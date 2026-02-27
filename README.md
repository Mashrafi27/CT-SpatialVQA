<p align="center">
    <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" align="center" width="30%">
</p>
<p align="center"><h1 align="center">CT-SpatialVQA</h1></p>
<p align="center">
	<em><code>❯ REPLACE-ME</code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/Mashrafi27/MICCAI2026-3DMedVLMS?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/Mashrafi27/MICCAI2026-3DMedVLMS?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/Mashrafi27/MICCAI2026-3DMedVLMS?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/Mashrafi27/MICCAI2026-3DMedVLMS?style=default&color=0080ff" alt="repo-language-count">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>

##  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Testing](#-testing)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

##  Overview

<code>❯ REPLACE-ME</code>

---

##  Features

<code>❯ REPLACE-ME</code>

---

##  Project Structure

```sh
└── MICCAI2026-3DMedVLMS/
    ├── Figures
    │   ├── category_radar_plot.pdf
    │   ├── ct_qa_pred_1.png
    │   ├── ct_rep_qa_1.png
    │   ├── ct_spatial_vqa_pipeline.png
    │   └── qa_pred_dist.png
    ├── MICCAI2026.pdf
    ├── QA_generation
    │   ├── README.md
    │   ├── filter_qa_pairs.py
    │   ├── human_eval
    │   ├── llm_judge.py
    │   ├── qa_generation.py
    │   ├── qa_to_jsonl.py
    │   ├── spatial_qa_filtered_full.json
    │   ├── spatial_qa_output_full.json
    │   └── validation_reports_output.json
    ├── README.md
    ├── README_3D_VLM_Spatial.md
    ├── benchmarking
    │   ├── README.md
    │   ├── eval_scripts
    │   ├── inference
    │   ├── inputs
    │   ├── preprocess
    │   └── reports
    └── dataset
        ├── ct_spatialvqa
        └── download_ctrate_dataset.py
```


###  Project Index
<details open>
	<summary><b><code>MICCAI2026-3DMEDVLMS/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			</table>
		</blockquote>
	</details>
	<details> <!-- dataset Submodule -->
		<summary><b>dataset</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/dataset/download_ctrate_dataset.py'>download_ctrate_dataset.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
			<details>
				<summary><b>ct_spatialvqa</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/dataset/ct_spatialvqa/spatial_qa_filtered_full.json'>spatial_qa_filtered_full.json</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/dataset/ct_spatialvqa/spatial_qa_filtered_full_nifti.jsonl'>spatial_qa_filtered_full_nifti.jsonl</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/dataset/ct_spatialvqa/spatial_qa_filtered_full_cases.jsonl'>spatial_qa_filtered_full_cases.jsonl</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- benchmarking Submodule -->
		<summary><b>benchmarking</b></summary>
		<blockquote>
			<details>
				<summary><b>preprocess</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/preprocess/preprocess_med3dvlm.py'>preprocess_med3dvlm.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/preprocess/preprocess_medgemma.py'>preprocess_medgemma.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/preprocess/preprocess_merlin.py'>preprocess_merlin.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/preprocess/preprocess_medevalkit.py'>preprocess_medevalkit.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/preprocess/preprocess_m3d.py'>preprocess_m3d.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/preprocess/preprocess_radfm.py'>preprocess_radfm.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/preprocess/preprocess_ctchat.py'>preprocess_ctchat.py</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>eval_scripts</b></summary>
				<blockquote>
					<details>
						<summary><b>scripts</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/eval_scripts/scripts/evaluate_with_qwen.py'>evaluate_with_qwen.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/eval_scripts/scripts/build_category_performance.py'>build_category_performance.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/eval_scripts/scripts/plot_answer_length_distributions.py'>plot_answer_length_distributions.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/eval_scripts/scripts/analyze_ct_spatialvqa.py'>analyze_ct_spatialvqa.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/eval_scripts/scripts/evaluate_with_gemini.py'>evaluate_with_gemini.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/eval_scripts/scripts/compute_llm_jury_accuracy.py'>compute_llm_jury_accuracy.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/eval_scripts/scripts/evaluate_text_metrics_all.py'>evaluate_text_metrics_all.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/eval_scripts/scripts/fix_gemini_parse_errors.py'>fix_gemini_parse_errors.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/eval_scripts/scripts/evaluate_with_gpt.py'>evaluate_with_gpt.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/eval_scripts/scripts/evaluate_text_metrics.py'>evaluate_text_metrics.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/eval_scripts/scripts/build_correctness_matrix.py'>build_correctness_matrix.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>spatial_categories</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/eval_scripts/spatial_categories/spatial_qa_filtered_full_with_categories.json'>spatial_qa_filtered_full_with_categories.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/eval_scripts/spatial_categories/spatial_qa_filtered_full.json'>spatial_qa_filtered_full.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/eval_scripts/spatial_categories/get_qa_pairs_categories.py'>get_qa_pairs_categories.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
			<details>
				<summary><b>reports</b></summary>
				<blockquote>
					<details>
						<summary><b>predictions</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/predictions/medevalkit_predictions_full.jsonl'>medevalkit_predictions_full.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/predictions/med3dvlm_predictions_full.jsonl'>med3dvlm_predictions_full.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/predictions/radfm_predictions_full.jsonl'>radfm_predictions_full.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/predictions/medgemma_predictions_full.jsonl'>medgemma_predictions_full.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/predictions/merlin_predictions_full.jsonl'>merlin_predictions_full.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/predictions/ctchat_predictions_full.jsonl'>ctchat_predictions_full.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/predictions/m3d_predictions_full.jsonl'>m3d_predictions_full.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/predictions/vila_m3_vista3d_predictions_full.jsonl'>vila_m3_vista3d_predictions_full.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>plots</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/plots/answer_length_distributions_data.json'>answer_length_distributions_data.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>metrics</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/metrics/llm_jury_accuracy.json'>llm_jury_accuracy.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/metrics/text_metrics_summary.json'>text_metrics_summary.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>analysis</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/analysis/dataset_counts.txt'>dataset_counts.txt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/analysis/ct_spatialvqa_analysis.json'>ct_spatialvqa_analysis.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>llm_eval</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/medgemma_qwen_eval.json'>medgemma_qwen_eval.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/radfm_qwen_eval.json'>radfm_qwen_eval.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/merlin_gpt_eval.json'>merlin_gpt_eval.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/ctchat_gemini2.5_eval_full.json'>ctchat_gemini2.5_eval_full.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/m3d_gemini2.5_eval_full.json'>m3d_gemini2.5_eval_full.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/merlin_gemini2.5_eval_full.json'>merlin_gemini2.5_eval_full.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/medevalkit_qwen_eval.json'>medevalkit_qwen_eval.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/medevalkit_gpt_eval.json'>medevalkit_gpt_eval.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/med3dvlm_qwen_eval.json'>med3dvlm_qwen_eval.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/vila_m3_vista3d_gemini2.5_eval_full.json'>vila_m3_vista3d_gemini2.5_eval_full.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/radfm_gemini2.5_eval_full.json'>radfm_gemini2.5_eval_full.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/medgemma_gemini2.5_eval_full.json'>medgemma_gemini2.5_eval_full.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/radfm_gpt_eval.json'>radfm_gpt_eval.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/vila_m3_vista3d_qwen_eval.json'>vila_m3_vista3d_qwen_eval.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/m3d_gpt_eval.json'>m3d_gpt_eval.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/med3dvlm_gpt_eval.json'>med3dvlm_gpt_eval.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/vila_m3_vista3d_gpt_eval.json'>vila_m3_vista3d_gpt_eval.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/merlin_qwen_eval.json'>merlin_qwen_eval.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/medgemma_gpt_eval.json'>medgemma_gpt_eval.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/m3d_qwen_eval.json'>m3d_qwen_eval.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/ctchat_qwen_eval.json'>ctchat_qwen_eval.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/medevalkit_gemini2.5_eval_full.json'>medevalkit_gemini2.5_eval_full.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/ctchat_gpt_eval.json'>ctchat_gpt_eval.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/reports/llm_eval/med3dvlm_gemini2.5_eval_full.json'>med3dvlm_gemini2.5_eval_full.json</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
			<details>
				<summary><b>inputs</b></summary>
				<blockquote>
					<details>
						<summary><b>merlin</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inputs/merlin/spatial_qa_filtered_full_nifti_merlin.jsonl'>spatial_qa_filtered_full_nifti_merlin.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inputs/merlin/spatial_qa_filtered_full_cases_merlin.jsonl'>spatial_qa_filtered_full_cases_merlin.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>medevalkit</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inputs/medevalkit/spatial_qa_filtered_full_nifti_medevalkit.jsonl'>spatial_qa_filtered_full_nifti_medevalkit.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inputs/medevalkit/spatial_qa_filtered_full_cases_medevalkit.jsonl'>spatial_qa_filtered_full_cases_medevalkit.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>ctchat</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inputs/ctchat/spatial_qa_filtered_full_nifti_ctchat.jsonl'>spatial_qa_filtered_full_nifti_ctchat.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>radfm</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inputs/radfm/spatial_qa_filtered_full_nifti_radfm.jsonl'>spatial_qa_filtered_full_nifti_radfm.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inputs/radfm/spatial_qa_filtered_full_cases_radfm.jsonl'>spatial_qa_filtered_full_cases_radfm.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>med3dvlm</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inputs/med3dvlm/spatial_qa_filtered_full_nifti_med3dvlm.jsonl'>spatial_qa_filtered_full_nifti_med3dvlm.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inputs/med3dvlm/spatial_qa_filtered_full_cases_med3dvlm.jsonl'>spatial_qa_filtered_full_cases_med3dvlm.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>ctchat_basic</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inputs/ctchat_basic/spatial_qa_filtered_full_nifti_ctchat_basic.jsonl'>spatial_qa_filtered_full_nifti_ctchat_basic.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inputs/ctchat_basic/spatial_qa_filtered_full_cases_ctchat_basic.jsonl'>spatial_qa_filtered_full_cases_ctchat_basic.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>vila_m3</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inputs/vila_m3/spatial_qa_filtered_full_nifti_vila_m3.jsonl'>spatial_qa_filtered_full_nifti_vila_m3.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inputs/vila_m3/spatial_qa_filtered_full_cases_vila_m3.jsonl'>spatial_qa_filtered_full_cases_vila_m3.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>m3d</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inputs/m3d/spatial_qa_filtered_full_nifti_m3d.jsonl'>spatial_qa_filtered_full_nifti_m3d.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inputs/m3d/spatial_qa_filtered_full_cases_m3d.jsonl'>spatial_qa_filtered_full_cases_m3d.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>medgemma</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inputs/medgemma/spatial_qa_filtered_full_nifti_medgemma896.jsonl'>spatial_qa_filtered_full_nifti_medgemma896.jsonl</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
				</blockquote>
			</details>
			<details>
				<summary><b>inference</b></summary>
				<blockquote>
					<details>
						<summary><b>ct-chat</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/ct-chat/encode_embeddings.py'>encode_embeddings.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/ct-chat/requirements.txt'>requirements.txt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/ct-chat/run_custom_eval.py'>run_custom_eval.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>merlin</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/merlin/requirements.txt'>requirements.txt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/merlin/run_custom_eval.py'>run_custom_eval.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>vila-m3</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/vila-m3/run_vista3d_eval.py'>run_vista3d_eval.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/vila-m3/run_vista3d_loop.sh'>run_vista3d_loop.sh</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/vila-m3/requirements.txt'>requirements.txt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/vila-m3/preprocess_resample_nifti.py'>preprocess_resample_nifti.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>medevalkit</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/medevalkit/requirements.txt'>requirements.txt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/medevalkit/run_custom_eval.py'>run_custom_eval.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>radfm</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/radfm/requirements.txt'>requirements.txt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/radfm/preprocess_radfm.py'>preprocess_radfm.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/radfm/run_custom_eval.py'>run_custom_eval.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>med3dvlm</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/med3dvlm/preprocess_ctrate.py'>preprocess_ctrate.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/med3dvlm/requirements.txt'>requirements.txt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/med3dvlm/run_custom_eval.py'>run_custom_eval.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/med3dvlm/play.ipynb'>play.ipynb</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>m3d</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/m3d/requirements.txt'>requirements.txt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/m3d/run_custom_eval.py'>run_custom_eval.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
						</blockquote>
					</details>
					<details>
						<summary><b>medgemma</b></summary>
						<blockquote>
							<table>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/medgemma/requirements.txt'>requirements.txt</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							<tr>
								<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/medgemma/run_custom_eval.py'>run_custom_eval.py</a></b></td>
								<td><code>❯ REPLACE-ME</code></td>
							</tr>
							</table>
							<details>
								<summary><b>notebooks</b></summary>
								<blockquote>
									<table>
									<tr>
										<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/benchmarking/inference/medgemma/notebooks/medgemma_ct_ourdata.ipynb'>medgemma_ct_ourdata.ipynb</a></b></td>
										<td><code>❯ REPLACE-ME</code></td>
									</tr>
									</table>
								</blockquote>
							</details>
						</blockquote>
					</details>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- QA_generation Submodule -->
		<summary><b>QA_generation</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/QA_generation/validation_reports_output.json'>validation_reports_output.json</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/QA_generation/spatial_qa_filtered_full.json'>spatial_qa_filtered_full.json</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/QA_generation/qa_generation.py'>qa_generation.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/QA_generation/spatial_qa_output_full.json'>spatial_qa_output_full.json</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/QA_generation/filter_qa_pairs.py'>filter_qa_pairs.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/QA_generation/llm_judge.py'>llm_judge.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/QA_generation/qa_to_jsonl.py'>qa_to_jsonl.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
			<details>
				<summary><b>human_eval</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/QA_generation/human_eval/progress.json'>progress.json</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/QA_generation/human_eval/users.json'>users.json</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/QA_generation/human_eval/gemini_human_disagreements.json'>gemini_human_disagreements.json</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/QA_generation/human_eval/summary.json'>summary.json</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/QA_generation/human_eval/assignments.json'>assignments.json</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/master/QA_generation/human_eval/summary.txt'>summary.txt</a></b></td>
						<td><code>❯ REPLACE-ME</code></td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
</details>

---
##  Getting Started

###  Prerequisites

Before getting started with MICCAI2026-3DMedVLMS, ensure your runtime environment meets the following requirements:

- **Programming Language:** Error detecting primary_language: {'py': 36, 'json': 39, 'jsonl': 26, 'txt': 10, 'sh': 1, 'ipynb': 2}
- **Package Manager:** Pip


###  Installation

Install MICCAI2026-3DMedVLMS using one of the following methods:

**Build from source:**

1. Clone the MICCAI2026-3DMedVLMS repository:
```sh
❯ git clone https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS
```

2. Navigate to the project directory:
```sh
❯ cd MICCAI2026-3DMedVLMS
```

3. Install the project dependencies:


**Using `pip`** &nbsp; [<img align="center" src="" />]()

```sh
❯ echo 'INSERT-INSTALL-COMMAND-HERE'
```




###  Usage
Run MICCAI2026-3DMedVLMS using the following command:
**Using `pip`** &nbsp; [<img align="center" src="" />]()

```sh
❯ echo 'INSERT-RUN-COMMAND-HERE'
```


###  Testing
Run the test suite using the following command:
**Using `pip`** &nbsp; [<img align="center" src="" />]()

```sh
❯ echo 'INSERT-TEST-COMMAND-HERE'
```


---
##  Project Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

##  Contributing

- **💬 [Join the Discussions](https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/issues)**: Submit bugs found or log feature requests for the `MICCAI2026-3DMedVLMS` project.
- **💡 [Submit Pull Requests](https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/Mashrafi27/MICCAI2026-3DMedVLMS
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/Mashrafi27/MICCAI2026-3DMedVLMS/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=Mashrafi27/MICCAI2026-3DMedVLMS">
   </a>
</p>
</details>

---

##  License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

##  Acknowledgments

- List any resources, contributors, inspiration, etc. here.

---