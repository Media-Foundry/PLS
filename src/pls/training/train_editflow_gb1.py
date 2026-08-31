"""Matched-query-budget GB1 value-KD and EditFlow proof-of-concept trainer."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from torch.utils.tensorboard import SummaryWriter

from pls.editflow.graph import exact_optimization_regret
from pls.editflow.hamming import hamming_distance, node_neighbors, variants_from_tokens
from pls.editflow.metrics import mutation_field_metrics
from pls.editflow.objective import editflow_distillation_loss
from pls.editflow.student import EditPotentialStudent


def connected_query_nodes(measured: np.ndarray, start: int, budget: int, seed: int, *, alphabet_size: int = 20, length: int = 4) -> np.ndarray:
    """Select a value-blind connected node set with fixed random priorities."""
    if not measured[start]:
        raise ValueError("query anchor must be experimentally measured")
    if budget < 1 or budget > int(measured.sum()):
        raise ValueError("invalid query budget")
    rng = np.random.default_rng(seed)
    selected, discovered = [int(start)], {int(start)}
    frontier: list[tuple[float, int]] = []

    def expand(node: int) -> None:
        for neighbor in node_neighbors(node, alphabet_size, length):
            neighbor = int(neighbor)
            if measured[neighbor] and neighbor not in discovered:
                discovered.add(neighbor)
                heapq.heappush(frontier, (float(rng.random()), neighbor))

    expand(start)
    while len(selected) < budget:
        if not frontier:
            raise RuntimeError("measured GB1 graph frontier was exhausted")
        _, node = heapq.heappop(frontier)
        selected.append(node);expand(node)
    return np.asarray(selected, dtype=np.int64)


def closed_local_edges(nodes: np.ndarray, *, alphabet_size: int = 20, length: int = 4) -> np.ndarray:
    local = {int(node): index for index, node in enumerate(nodes)}
    sources, targets = [], []
    for source_local, source_global in enumerate(nodes):
        for target_global in node_neighbors(int(source_global), alphabet_size, length):
            target_local = local.get(int(target_global))
            if target_local is not None and source_local < target_local:
                sources.append(source_local);targets.append(target_local)
    return np.asarray((sources, targets), dtype=np.int64)


def evaluation_edges(tokens: np.ndarray, measured: np.ndarray, count: int, salt: str):
    variants = variants_from_tokens(tokens, "ACDEFGHIKLMNPQRSTVWY")
    ranked = sorted(
        (hashlib.sha256(f"{salt}:{variant}".encode()).digest(), index)
        for index, variant in enumerate(variants) if measured[index]
    )[:count]
    anchors = np.asarray([index for _, index in ranked], dtype=np.int64)
    sources, targets, groups = [], [], []
    for group, source in enumerate(anchors):
        for target in node_neighbors(int(source), 20, 4):
            if measured[target]:
                sources.append(int(source));targets.append(int(target));groups.append(group)
    return np.asarray((sources, targets), dtype=np.int64), np.asarray(groups, dtype=np.int64)


def batched_predict(model, tokens: torch.Tensor, device, batch_size: int) -> np.ndarray:
    model.eval();predictions=[]
    with torch.inference_mode():
        for start in range(0, len(tokens), batch_size):
            batch = tokens[start:start + batch_size].to(device, non_blocking=True)
            predictions.append(model(batch, torch.ones_like(batch, dtype=torch.bool)).float().cpu().numpy())
    return np.concatenate(predictions)


def main() -> None:
    parser = argparse.ArgumentParser();parser.add_argument("--config", type=Path, required=True);parser.add_argument("--run-dir", type=Path, required=True);arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text());data_config=config["data"];model_config=config["model"];training=config["training"]
    if config.get("evaluate_test", False):parser.error("test evaluation is permanently disabled")
    if os.environ.get("HIP_VISIBLE_DEVICES") != str(training["hip_device"]):parser.error("HIP device mismatch")
    seed=int(training["seed"]);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    landscape=np.load(data_config["landscape"]);raw_tokens=landscape["tokens"].astype(np.int64);tokens_np=raw_tokens+1;fitness=landscape["fitness"].astype(np.float64);measured=landscape["is_measured"].astype(bool);wild_type="VDGV";alphabet="ACDEFGHIKLMNPQRSTVWY";wild_tokens=np.asarray([alphabet.index(value) for value in wild_type]);wild_index=int(np.ravel_multi_index(tuple(wild_tokens),(20,)*4))
    queried=connected_query_nodes(measured,wild_index,int(data_config["query_budget"]),int(data_config["query_seed"]));query_edges=closed_local_edges(queried);query_hash=hashlib.sha256(queried.astype("<i8").tobytes()).hexdigest();target_mean=float(fitness[queried].mean());target_std=float(fitness[queried].std());target_std=max(target_std,1e-6);teacher=torch.tensor((fitness[queried]-target_mean)/target_std,dtype=torch.float32);tokens=torch.from_numpy(tokens_np);query_tokens=tokens[queried];device=torch.device("cuda:0")
    model=EditPotentialStudent(model_config["dimension"],model_config["layers"],model_config["heads"],model_config["dropout"],4).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=training["learning_rate"],weight_decay=training["weight_decay"],fused=training.get("fused_optimizer",False));writer=SummaryWriter(arguments.run_dir/"tensorboard");history=[];best=float("inf")
    query_tokens=query_tokens.to(device);teacher=teacher.to(device);query_edges_tensor=torch.from_numpy(query_edges).to(device);queried_mask=torch.ones(len(queried),dtype=torch.bool,device=device)
    for epoch in range(1,int(training["epochs"])+1):
        started=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True);prediction=model(query_tokens,torch.ones_like(query_tokens,dtype=torch.bool));report=editflow_distillation_loss(prediction,teacher,query_edges_tensor,queried_mask,value_weight=1.,edge_weight=float(training["edge_weight"]),loss=training.get("loss","mse"));report.total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float(training.get("max_gradient_norm",5.)));optimizer.step();row={"epoch":epoch,"total_loss":float(report.total.detach()),"value_loss":float(report.value.detach()),"edge_loss":float(report.edge.detach()),"seconds":time.monotonic()-started};history.append(row);writer.add_scalar("training/total_loss",row["total_loss"],epoch);writer.add_scalar("training/value_loss",row["value_loss"],epoch);writer.add_scalar("training/edge_loss",row["edge_loss"],epoch)
        if row["total_loss"]<best:best=row["total_loss"];torch.save({"model":model.state_dict(),"epoch":epoch,"config":config},arguments.run_dir/"checkpoints"/"best.pt")
        if epoch==1 or epoch%int(training.get("log_every",25))==0:print(json.dumps(row),flush=True)
    writer.close();(arguments.run_dir/"history.json").write_text(json.dumps(history,indent=2)+"\n");state=torch.load(arguments.run_dir/"checkpoints"/"best.pt",map_location=device,weights_only=False);model.load_state_dict(state["model"]);normalized=batched_predict(model,tokens,device,int(training.get("inference_batch_size",8192)));prediction=normalized*target_std+target_mean
    measured_truth=fitness[measured];measured_prediction=prediction[measured];value_metrics={"measured_nodes":int(measured.sum()),"r2":float(r2_score(measured_truth,measured_prediction)),"pearson":float(pearsonr(measured_truth,measured_prediction).statistic),"rmse":float(np.sqrt(np.mean(np.square(measured_prediction-measured_truth))))}
    field_edges,edge_groups=evaluation_edges(raw_tokens,measured,int(data_config["evaluation_anchors"]),data_config["evaluation_salt"]);edge_metrics=mutation_field_metrics(fitness,prediction,field_edges,edge_groups,top_k=int(data_config.get("top_k",10)))
    distances=hamming_distance(raw_tokens,wild_tokens);regret={};
    for radius in data_config["edit_radii"]:
        candidates=np.flatnonzero(measured&(distances<=int(radius)));regret[str(radius)]=exact_optimization_regret(fitness,prediction,candidates)
    query_budget={"unique_queried_nodes":int(len(queried)),"closed_edges":int(query_edges.shape[1]),"queried_nodes_sha256":query_hash,"query_seed":int(data_config["query_seed"]),"selection":"value_blind_connected_random_frontier","same_node_budget_required":True}
    for name,value in (("value_metrics.json",value_metrics),("edge_metrics.json",edge_metrics),("regret_metrics.json",regret),("query_budget.json",query_budget)):(arguments.run_dir/name).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"best_epoch":state["epoch"],"value":value_metrics,"edge":edge_metrics,"regret":regret,"query_budget":query_budget,"test_evaluated":False}),flush=True)


if __name__=="__main__":main()
