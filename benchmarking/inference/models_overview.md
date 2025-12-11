# Model Overview

This file mirrors the spreadsheet contents for quick reference when scripting inference harnesses.

## Med3DVLM
- Image-text retrieval (cross-modal search)
- Radiology report generation
- Visual question answering (open/closed)
- Spatial focus: positions, depth, distance

## MS-VLM
- Radiology report generation (CT-RATE chest CT + rectal MRI)
- Multi-plane, multi-phase interpretation
- Foundation for future 3D medical VQA

## Merlin
- Classification (findings, phenotypes)
- Retrieval (text-image alignment)
- Prediction (long-term disease risk)
- Generation (radiology reports)
- Segmentation (organ-level voxels)

## BrainMD
- Disease diagnosis (brain tumor classification)
- Medical VQA (tumors, anatomy, clinical context)
- Report generation/interpretation with MRI + text records

## DeepTumorVQA
1. Recognition: organ/lesion detection, counting, typing, fatty liver recognition
2. Measurement: organ/lesion volumes, HU, comparisons
3. Visual reasoning: adjacency, inter-segment comparisons, distribution across anatomy
4. Medical reasoning: staging, resectability, diagnosis-level reasoning

## VILA-M3
- Image segmentation
- Abnormality classification
- Medical report generation
- Visual question answering

## E3D-GPT
- Report generation from 3D imaging
- Visual question answering (interactive diagnostic Q&A)
- Disease diagnosis

## FILA (FVLM)
- Multimodal medical reasoning per OpenReview submission
