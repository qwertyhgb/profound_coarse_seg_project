#!/usr/bin/env python
"""Sweep coarse proposal post-processing rules and export Stage-2 prompts."""
from __future__ import annotations
import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any


def _parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(',') if x.strip()]


def _load_cases(component_json: Path) -> dict[str, Any]:
    if not component_json.is_file():
        raise FileNotFoundError(
            f"Component detail file not found: {component_json}. "
            "Run evaluate_coarse_proposals.py with --save-component-json first."
        )
    return json.loads(component_json.read_text())


def _rank_value(component: dict[str, Any], rank_by: str) -> float:
    if rank_by == 'max_probability':
        return float(component.get('max_probability', 0.0))
    if rank_by == 'mean_probability':
        return float(component.get('mean_probability', 0.0))
    if rank_by == 'component_volume':
        return float(component.get('voxels', 0.0))
    if rank_by == 'box_volume':
        return float(component.get('box_volume', 0.0))
    if rank_by == 'maxprob_x_volume':
        return float(component.get('max_probability', 0.0)) * float(component.get('voxels', 0.0))
    raise ValueError(f"Unsupported rank_by: {rank_by}")


def _filter_components(
    components: list[dict[str, Any]],
    min_component_size: int,
    min_max_probability: float,
    top_k: int,
    rank_by: str,
) -> list[dict[str, Any]]:
    kept = [
        c for c in components
        if int(c.get('voxels', 0)) >= min_component_size
        and float(c.get('max_probability', 0.0)) >= min_max_probability
    ]
    kept.sort(key=lambda c: _rank_value(c, rank_by), reverse=True)
    if top_k > 0:
        kept = kept[:top_k]
    return kept


def _gt_hit(gt: dict[str, Any], kept_components: list[dict[str, Any]]) -> bool:
    # During validation analysis, component extraction stores whether a pred
    # component overlaps any GT lesion. It does not store component-to-lesion
    # identity, so a GT is hit when any retained TP component exists. For PI-CAI
    # most positive cases have one lesion; multi-lesion recall remains a coarse
    # validation estimate here.
    if not kept_components:
        return False
    return any(bool(c.get('overlaps_gt', False)) for c in kept_components)


def _evaluate_strategy(cases: dict[str, Any], min_size: int, min_prob: float, top_k: int, rank_by: str) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    totals = {
        'cases': 0,
        'positive_cases': 0,
        'total_gt_lesions': 0,
        'hit_gt_lesions': 0,
        'pred_components': 0,
        'tp_pred_components': 0,
        'fp_pred_components': 0,
        'prompt_voxels': 0,
    }
    kept_by_case: dict[str, list[dict[str, Any]]] = {}
    for case_id, payload in cases.items():
        totals['cases'] += 1
        gt_components = payload.get('gt_components', [])
        pred_components = payload.get('pred_components', [])
        kept = _filter_components(pred_components, min_size, min_prob, top_k, rank_by)
        kept_by_case[case_id] = kept

        n_gt = len(gt_components)
        totals['positive_cases'] += int(n_gt > 0)
        totals['total_gt_lesions'] += n_gt
        hit_case_gt = 0
        if n_gt > 0:
            # Exact component-to-GT identity is not saved, but retained TP
            # proposals tell us the case has a usable prompt candidate. Cap by
            # n_gt to avoid over-counting multi-component predictions.
            hit_case_gt = min(n_gt, sum(1 for c in kept if bool(c.get('overlaps_gt', False))))
            # If one retained TP component exists in a single-lesion case, it is hit.
            if n_gt == 1 and any(bool(c.get('overlaps_gt', False)) for c in kept):
                hit_case_gt = 1
        totals['hit_gt_lesions'] += hit_case_gt
        totals['pred_components'] += len(kept)
        totals['tp_pred_components'] += sum(1 for c in kept if bool(c.get('overlaps_gt', False)))
        totals['fp_pred_components'] += sum(1 for c in kept if not bool(c.get('overlaps_gt', False)))
        totals['prompt_voxels'] += sum(int(c.get('voxels', 0)) for c in kept)

    cases_n = max(totals['cases'], 1)
    pred_n = max(totals['pred_components'], 1)
    summary = {
        **totals,
        'min_component_size': min_size,
        'min_max_probability': min_prob,
        'top_k_per_case': top_k,
        'rank_by': rank_by,
        'lesion_recall': totals['hit_gt_lesions'] / totals['total_gt_lesions'] if totals['total_gt_lesions'] else 0.0,
        'candidates_per_case': totals['pred_components'] / cases_n,
        'fp_per_case': totals['fp_pred_components'] / cases_n,
        'component_precision': totals['tp_pred_components'] / pred_n if totals['pred_components'] else 0.0,
        'mean_prompt_voxels': totals['prompt_voxels'] / cases_n,
    }
    return summary, kept_by_case


def _strategy_score(row: dict[str, Any], target_recall: float, fp_weight: float, candidate_weight: float) -> float:
    recall = float(row['lesion_recall'])
    recall_shortfall = max(0.0, target_recall - recall)
    return recall - 2.0 * recall_shortfall - fp_weight * float(row['fp_per_case']) - candidate_weight * float(row['candidates_per_case'])


def _choose_best(rows: list[dict[str, Any]], target_recall: float, fp_weight: float, candidate_weight: float) -> dict[str, Any]:
    feasible = [r for r in rows if float(r['lesion_recall']) >= target_recall]
    pool = feasible or rows
    return max(pool, key=lambda r: _strategy_score(r, target_recall, fp_weight, candidate_weight))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _box_center(box: list[int]) -> list[float]:
    z0, z1, y0, y1, x0, x1 = [float(v) for v in box]
    return [(z0 + z1 - 1.0) / 2.0, (y0 + y1 - 1.0) / 2.0, (x0 + x1 - 1.0) / 2.0]


def _export_prompts(output_dir: Path, kept_by_case: dict[str, list[dict[str, Any]]], strategy: dict[str, Any], include_gt_hit: bool) -> None:
    prompt_rows = []
    prompt_json = []
    for case_id, components in kept_by_case.items():
        for rank, comp in enumerate(components, start=1):
            box = [int(v) for v in comp['box_zyxzyx']]
            center = _box_center(box)
            row = {
                'case_id': case_id,
                'proposal_rank': rank,
                'component_id': int(comp.get('component_id', rank)),
                'z0': box[0], 'z1': box[1], 'y0': box[2], 'y1': box[3], 'x0': box[4], 'x1': box[5],
                'center_z': center[0], 'center_y': center[1], 'center_x': center[2],
                'component_voxels': int(comp.get('voxels', 0)),
                'box_volume': int(comp.get('box_volume', 0)),
                'max_probability': float(comp.get('max_probability', 0.0)),
                'mean_probability': float(comp.get('mean_probability', 0.0)),
                'rank_by': strategy['rank_by'],
                'min_component_size': strategy['min_component_size'],
                'min_max_probability': strategy['min_max_probability'],
                'top_k_per_case': strategy['top_k_per_case'],
            }
            if include_gt_hit:
                row['overlaps_gt'] = bool(comp.get('overlaps_gt', False))
            prompt_rows.append(row)
            prompt_json.append(row)
    _write_csv(output_dir / 'coarse_prompts.csv', prompt_rows)
    (output_dir / 'coarse_prompts.json').write_text(json.dumps(prompt_json, indent=2))


def _write_report(path: Path, rows: list[dict[str, Any]], best: dict[str, Any], target_recall: float) -> None:
    best_first = [best]
    seen = {(best['min_component_size'], best['min_max_probability'], best['top_k_per_case'], best['rank_by'])}
    ranked = sorted(rows, key=lambda r: r.get('selection_score', 0.0), reverse=True)
    top_rows = best_first + [
        r for r in ranked
        if (r['min_component_size'], r['min_max_probability'], r['top_k_per_case'], r['rank_by']) not in seen
    ][:11]
    lines = [
        '# Proposal Post-processing Sweep',
        '',
        f'- Target lesion recall: {target_recall:.4f}',
        f"- Best strategy: min_component_size={best['min_component_size']}, min_max_probability={best['min_max_probability']}, top_k={best['top_k_per_case']}, rank_by={best['rank_by']}",
        f"- Best lesion recall: {best['lesion_recall']:.4f} ({best['hit_gt_lesions']}/{best['total_gt_lesions']})",
        f"- Best candidates/case: {best['candidates_per_case']:.4f}",
        f"- Best fp/case: {best['fp_per_case']:.4f}",
        f"- Best component precision: {best['component_precision']:.4f}",
        '',
        '| min_size | min_prob | top_k | rank_by | lesion_recall | candidates/case | fp/case | precision |',
        '|---:|---:|---:|---|---:|---:|---:|---:|',
    ]
    for row in top_rows:
        lines.append(
            f"| {row['min_component_size']} | {row['min_max_probability']:.2f} | {row['top_k_per_case']} | {row['rank_by']} | "
            f"{row['lesion_recall']:.4f} | {row['candidates_per_case']:.4f} | {row['fp_per_case']:.4f} | {row['component_precision']:.4f} |"
        )
    lines.append('')
    lines.append('Use `coarse_prompts.csv/json` as the first Stage-2 box and center-point prompt file. `overlaps_gt` is for validation analysis only and must not be used at test time.')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--component-json', default='outputs/coarse_score_es_3090/fold_0/inference/proposal_reports/component_details.json')
    parser.add_argument('--output-dir', default='outputs/coarse_score_es_3090/fold_0/inference/postprocess_sweep')
    parser.add_argument('--min-component-sizes', default='2,10,20,50,100,200')
    parser.add_argument('--min-max-probabilities', default='0.0,0.25,0.5,0.7,0.9')
    parser.add_argument('--top-k', default='1,2,3,4,5,0', help='0 means keep all after filtering')
    parser.add_argument('--rank-by', default='max_probability,mean_probability,component_volume,maxprob_x_volume')
    parser.add_argument('--target-recall', type=float, default=0.90)
    parser.add_argument('--fp-weight', type=float, default=0.08)
    parser.add_argument('--candidate-weight', type=float, default=0.03)
    parser.add_argument('--include-gt-hit-in-prompts', action='store_true')
    args = parser.parse_args()

    cases = _load_cases(Path(args.component_json))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_sizes = _parse_ints(args.min_component_sizes)
    min_probs = _parse_floats(args.min_max_probabilities)
    top_ks = _parse_ints(args.top_k)
    rank_bys = [x.strip() for x in args.rank_by.split(',') if x.strip()]

    rows = []
    kept_cache = {}
    for min_size, min_prob, top_k, rank_by in itertools.product(min_sizes, min_probs, top_ks, rank_bys):
        summary, kept = _evaluate_strategy(cases, min_size, min_prob, top_k, rank_by)
        summary['selection_score'] = _strategy_score(summary, args.target_recall, args.fp_weight, args.candidate_weight)
        rows.append(summary)
        key = (min_size, min_prob, top_k, rank_by)
        kept_cache[key] = kept

    rows.sort(key=lambda r: r['selection_score'], reverse=True)
    best = _choose_best(rows, args.target_recall, args.fp_weight, args.candidate_weight)
    best_key = (best['min_component_size'], best['min_max_probability'], best['top_k_per_case'], best['rank_by'])
    best_kept = kept_cache[best_key]

    _write_csv(output_dir / 'postprocess_sweep.csv', rows)
    _write_report(output_dir / 'postprocess_summary.md', rows, best, args.target_recall)
    _export_prompts(output_dir, best_kept, best, include_gt_hit=args.include_gt_hit_in_prompts)
    (output_dir / 'best_strategy.json').write_text(json.dumps(best, indent=2))

    print(json.dumps(best, indent=2))
    print(f'Saved postprocess sweep and Stage-2 prompts to {output_dir}')


if __name__ == '__main__':
    main()
