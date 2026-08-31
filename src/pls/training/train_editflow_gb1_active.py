"""Sequential GB1 acquisition with uncertainty- or path-aware teacher queries."""

from __future__ import annotations

import argparse
import copy
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

from pls.editflow.acquisition import (AcquisitionBatch, ensemble_edge_uncertainty,
                                      frontier_node_acquisition)
from pls.editflow.graph import exact_design_regrets
from pls.editflow.hamming import (hamming_distance, node_neighbors,
                                  queried_nodes_sha256)
from pls.editflow.metrics import mutation_field_metrics
from pls.editflow.objective import editflow_distillation_loss
from pls.editflow.optimization import (bound_aware_frontier_acquisition,
                                       hybrid_query_budget,
                                       path_aware_frontier_acquisition)
from pls.editflow.student import EditPotentialStudent
from pls.training.train_editflow_gb1 import (batched_predict,
                                             closed_local_edges,
                                             connected_query_nodes,
                                             evaluation_edges)


def frontier_edges(queried_nodes, available: np.ndarray) -> np.ndarray:
    queried = frozenset(map(int, queried_nodes));sources=[];targets=[]
    for source in sorted(queried):
        for target in node_neighbors(source, 20, 4):
            target=int(target)
            if available[target] and target not in queried:
                sources.append(source);targets.append(target)
    return np.asarray((sources, targets), dtype=np.int64)


def fit_ensemble(tokens, fitness, queried, model_config, training, device):
    queried=np.asarray(sorted(queried),dtype=np.int64);local_edges=closed_local_edges(queried);target_mean=float(fitness[queried].mean());target_std=max(float(fitness[queried].std()),1e-6);teacher=torch.tensor((fitness[queried]-target_mean)/target_std,dtype=torch.float32,device=device);query_tokens=tokens[queried].to(device);mask=torch.ones_like(query_tokens,dtype=torch.bool);edge_index=torch.from_numpy(local_edges).to(device);queried_mask=torch.ones(len(queried),dtype=torch.bool,device=device);predictions=[];states=[];summaries=[]
    for member,seed in enumerate(training["ensemble_seeds"]):
        torch.manual_seed(int(seed));model=EditPotentialStudent(model_config["dimension"],model_config["layers"],model_config["heads"],model_config["dropout"],4).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=training["learning_rate"],weight_decay=training["weight_decay"],fused=training.get("fused_optimizer",False));best=float("inf");best_state=None;started=time.monotonic()
        for epoch in range(1,int(training["epochs_per_round"])+1):
            model.train();optimizer.zero_grad(set_to_none=True);values=model(query_tokens,mask);loss=editflow_distillation_loss(values,teacher,edge_index,queried_mask,value_weight=1.,edge_weight=float(training["edge_weight"]),loss=training.get("loss","mse"));loss.total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float(training.get("max_gradient_norm",5.)));optimizer.step()
            if float(loss.total.detach())<best:best=float(loss.total.detach());best_state={name:value.detach().cpu().clone() for name,value in model.state_dict().items()};best_epoch=epoch
        model.load_state_dict(best_state);normalized=batched_predict(model,tokens,device,int(training.get("inference_batch_size",8192)));predictions.append(normalized*target_std+target_mean);states.append(best_state);summaries.append({"member":member,"seed":int(seed),"best_epoch":best_epoch,"best_training_objective":best,"seconds":time.monotonic()-started})
    return np.asarray(predictions),states,summaries,int(local_edges.shape[1])


def evaluate(ensemble,fitness,measured,distances,field_edges,edge_groups,radii,top_k,queried):
    prediction=ensemble.mean(0);truth=fitness[measured];estimate=prediction[measured];value={"measured_nodes":int(measured.sum()),"r2":float(r2_score(truth,estimate)),"pearson":float(pearsonr(truth,estimate).statistic),"rmse":float(np.sqrt(np.mean(np.square(estimate-truth))))};edge=mutation_field_metrics(fitness,prediction,field_edges,edge_groups,top_k=top_k);regret={}
    for radius in radii:
        candidates=np.flatnonzero(measured&(distances<=int(radius)));regret[str(radius)]=exact_design_regrets(fitness,prediction,candidates,queried)
    return value,edge,regret


def uncertainty_acquisition(
    ensemble,
    queried,
    measured,
    budget,
    *,
    excluded_targets=(),
):
    """Acquire one-hop frontier nodes without expanding through this batch.

    ``excluded_targets`` is used when uncertainty fills a partially populated
    path-aware batch.  Those prospective nodes must not become sources until
    their teacher values have actually been purchased in the next round.
    """
    edges = frontier_edges(queried, measured)
    excluded = frozenset(map(int, excluded_targets))
    if excluded and edges.shape[1]:
        keep = ~np.isin(edges[1], np.fromiter(excluded, dtype=np.int64))
        edges = edges[:, keep]
    uncertainty = ensemble_edge_uncertainty(ensemble, edges)
    already_owned = set(map(int, queried)) | set(excluded)
    batch = frontier_node_acquisition(
        edges,
        uncertainty,
        np.ones_like(uncertainty),
        already_owned,
        budget,
    )
    return batch, edges


def frontier_policy_acquisition(
    ensemble,
    queried,
    measured,
    budget,
    policy,
    rng: np.random.Generator,
    *,
    beta: float = 1.0,
    excluded_targets=(),
):
    """Standard equal-cost frontier baselines over unique target nodes."""
    if policy not in {"random", "greedy", "ucb", "thompson"}:
        raise ValueError("policy must be random, greedy, ucb, or thompson")
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    values = np.asarray(ensemble, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("ensemble must have shape [members, nodes]")
    edges = frontier_edges(queried, measured)
    excluded = frozenset(map(int, excluded_targets))
    targets = np.asarray(
        sorted(set(map(int, edges[1])) - set(map(int, queried)) - set(excluded)),
        dtype=np.int64,
    )
    if policy == "random":
        scores = rng.random(len(targets))
    elif policy == "greedy":
        scores = values[:, targets].mean(0)
    elif policy == "ucb":
        scores = values[:, targets].mean(0) + beta * values[:, targets].std(
            axis=0, ddof=1
        )
    else:
        member = int(rng.integers(values.shape[0]))
        scores = values[member, targets]
    ranked = sorted(
        zip(targets.tolist(), scores.tolist()), key=lambda item: (-item[1], item[0])
    )[:budget]
    batch = AcquisitionBatch(
        node_indices=np.asarray([node for node, _ in ranked], dtype=np.int64),
        scores=np.asarray([score for _, score in ranked], dtype=np.float64),
        candidate_edges=int(edges.shape[1]),
    )
    return batch, edges


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True);parser.add_argument("--run-dir",type=Path,required=True);arguments=parser.parse_args();config=json.loads(arguments.config.read_text());data_config=config["data"];model_config=config["model"];training=config["training"]
    if config.get("evaluate_test",False):parser.error("test evaluation is permanently disabled")
    if os.environ.get("HIP_VISIBLE_DEVICES")!=str(training["hip_device"]):parser.error("HIP device mismatch")
    seed=int(training["seed"]);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);device=torch.device("cuda:0");landscape=np.load(data_config["landscape"]);raw_tokens=landscape["tokens"].astype(np.int64);tokens=torch.from_numpy(raw_tokens+1);fitness=landscape["fitness"].astype(np.float64);measured=landscape["is_measured"].astype(bool);alphabet="ACDEFGHIKLMNPQRSTVWY";wild_tokens=np.asarray([alphabet.index(value) for value in "VDGV"]);anchor=int(np.ravel_multi_index(tuple(wild_tokens),(20,)*4));distances=hamming_distance(raw_tokens,wild_tokens);field_edges,edge_groups=evaluation_edges(raw_tokens,measured,int(data_config["evaluation_anchors"]),data_config["evaluation_salt"]);budgets=list(map(int,data_config["query_budgets"]));queried=set(map(int,connected_query_nodes(measured,anchor,budgets[0],int(data_config["initial_query_seed"]))));writer=SummaryWriter(arguments.run_dir/"tensorboard");history=[];rollouts=[];stages=[];final_states=None
    for round_index,budget in enumerate(budgets):
        if len(queried)!=budget:raise RuntimeError(f"query budget mismatch before round: {len(queried)} != {budget}")
        ensemble,states,training_summary,closed_edges=fit_ensemble(tokens,fitness,queried,model_config,training,device);value_metrics,edge_metrics,regret_metrics=evaluate(ensemble,fitness,measured,distances,field_edges,edge_groups,data_config["edit_radii"],int(data_config.get("top_k",10)),queried);stage={"round":round_index,"budget":budget,"queried_nodes_sha256":queried_nodes_sha256(queried),"closed_edges":closed_edges,"training":training_summary,"value":value_metrics,"edge":edge_metrics,"regret":regret_metrics};stages.append(stage);print(json.dumps(stage),flush=True);writer.add_scalar("budget_curve/r2",value_metrics["r2"],budget);writer.add_scalar("budget_curve/edge_spearman",edge_metrics["edge_spearman"],budget);writer.add_scalar("budget_curve/acquired_regret_radius_4",regret_metrics["4"]["acquired"]["regret"],budget);novel_regret=regret_metrics["4"]["novel_design"]["regret"];writer.add_scalar("budget_curve/campaign_regret_radius_4",regret_metrics["4"]["campaign"]["regret"],budget);writer.add_scalar("budget_curve/novel_design_regret_radius_4",novel_regret,budget) if novel_regret is not None else None;final_states=states
        if round_index==len(budgets)-1:break
        increment=budgets[round_index+1]-budget;mode=data_config["acquisition"]
        if mode=="path_aware":
            acquired=path_aware_frontier_acquisition(ensemble,queried,measured,anchor,increment,alphabet_size=20,length=4,steps=int(data_config["beam_steps"]),beam_width=int(data_config["beam_width"]),conservative_beta=float(data_config.get("conservative_beta",0)));selected=acquired.batch.node_indices.tolist();details={"mode":mode,"path_count":len(acquired.paths),"path_edges":int(acquired.path_edges.shape[1]),"path_selected":len(selected)}
        elif mode=="hybrid_path":
            targeted_budget=hybrid_query_budget(increment,float(data_config["path_fraction"]));acquired=path_aware_frontier_acquisition(ensemble,queried,measured,anchor,targeted_budget,alphabet_size=20,length=4,steps=int(data_config["beam_steps"]),beam_width=int(data_config["beam_width"]),conservative_beta=float(data_config.get("conservative_beta",0)));selected=acquired.batch.node_indices.tolist();details={"mode":mode,"path_fraction":float(data_config["path_fraction"]),"path_budget":targeted_budget,"exploration_budget":increment-targeted_budget,"path_count":len(acquired.paths),"path_edges":int(acquired.path_edges.shape[1]),"path_selected":len(selected)}
        elif mode=="bound_aware":
            acquired=bound_aware_frontier_acquisition(ensemble,queried,measured,anchor,increment,alphabet_size=20,length=4,steps=int(data_config["beam_steps"]),beam_width=int(data_config["beam_width"]),conservative_beta=float(data_config.get("conservative_beta",0)));selected=acquired.batch.node_indices.tolist();details={"mode":mode,"candidate_endpoints":len(acquired.candidate_endpoints),"bound_paths":len(acquired.selected_paths),"path_edges":int(acquired.path_edges.shape[1]),"bound_selected":len(selected),"mean_estimated_path_bound":float(acquired.estimated_path_bounds.mean()) if len(acquired.estimated_path_bounds) else 0.0}
        elif mode=="uncertainty":
            acquired,edges=uncertainty_acquisition(ensemble,queried,measured,increment);selected=acquired.node_indices.tolist();details={"mode":mode,"frontier_edges":int(edges.shape[1]),"uncertainty_selected":len(selected)}
        else:raise ValueError("acquisition must be path_aware, hybrid_path, bound_aware, or uncertainty")
        if len(selected)<increment:
            fill,edges=uncertainty_acquisition(
                ensemble,
                queried,
                measured,
                increment-len(selected),
                excluded_targets=selected,
            );selected.extend(fill.node_indices.tolist());details["uncertainty_fill"]=len(fill.node_indices);details["fill_frontier_edges"]=int(edges.shape[1])
        if len(selected)!=increment or set(selected)&queried:raise RuntimeError("acquisition did not purchase the exact new-node budget")
        queried.update(map(int,selected));details.update({"from_budget":budget,"to_budget":len(queried),"selected_nodes":list(map(int,selected))});rollouts.append(details)
    writer.close();torch.save({"members":final_states,"config":config,"queried_nodes":sorted(queried)},arguments.run_dir/"checkpoints"/"best.pt");(arguments.run_dir/"history.json").write_text(json.dumps(stages,indent=2)+"\n");(arguments.run_dir/"optimization_rollouts.json").write_text(json.dumps(rollouts,indent=2)+"\n");query_manifest={"schema":"PLS_EditFlow_queried_nodes_v1","node_indices":sorted(queried),"sha256":queried_nodes_sha256(queried),"oracle_values_included":False};(arguments.run_dir/"queried_nodes.json").write_text(json.dumps(query_manifest,indent=2,sort_keys=True)+"\n");query_budget={"acquisition":data_config["acquisition"],"budget_curve":budgets,"final_unique_queried_nodes":len(queried),"final_queried_nodes_sha256":query_manifest["sha256"],"rounds":len(budgets),"teacher_query_cost_unit":"unique measured node","test_evaluated":False};(arguments.run_dir/"query_budget.json").write_text(json.dumps(query_budget,indent=2,sort_keys=True)+"\n");final=stages[-1]
    for name,value in (("value_metrics.json",final["value"]),("edge_metrics.json",final["edge"]),("ranking_metrics.json",{key:value for key,value in final["edge"].items() if "kendall" in key or "recall" in key or "sign" in key}),("regret_metrics.json",final["regret"])):(arguments.run_dir/name).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"final":final,"query_budget":query_budget,"test_evaluated":False}),flush=True)


if __name__=="__main__":main()
