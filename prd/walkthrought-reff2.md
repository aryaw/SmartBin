# YOLOv26-seg Waste Instance Segmentation CMS Pipeline

## Objective

Update the existing CMS under `http://localhost:3000/raw` so the complete waste segmentation workflow is available through a simple, non-technical user interface.

Use `walkthrought.md` as the main project reference and `walkthrought-reff.md` as the technical workflow reference.

Before changing code, inspect the current frontend and backend implementation and answer through implementation evidence whether each required process already exists, partially exists, or is missing. Do not assume that a feature exists only because a route, button, component, or endpoint name exists. Verify that the frontend action is connected to a working backend process and that the process produces real artifacts and report data.

The required technical flow is:

```text
Explore Original Dataset Structure
Convert to YOLO-seg Format with Polygon Masks
Create data.yaml
Visualize Polygon Masks on Images
Train YOLOv26m-seg with Optimized Hyperparameters
Training Results and Curves
Validation and Test Evaluation with Box and Mask Metrics
Inference with Segmentation Masks and Labels
Run on Original Dataset Images with Recycling Advice
Export and Package for Deployment
Final Verification
```

The CMS must simplify this technical flow into four sidebar menus for non-technical users.

## Project-Wide Writing Rules

Apply these rules to all new or updated files in the project directory:

- Do not use em dash characters.
- Do not use arrow characters.
- Do not use emojis.
- Do not add numbers to sidebar menu labels.
- Do not fabricate metrics, test results, artifacts, statuses, or process completion.
- Do not copy Kaggle paths into the local application.
- Do not hardcode results from `walkthrought-reff.md`.
- Use actual local process output for all CMS reports.
- Keep labels understandable for non-technical users.
- Keep `walkthrought.md` concise and aligned with the implemented application.

## Dataset Location

The dataset already exists at:

`backend/dataset/raw`

Do not download the dataset.

Do not clone, fetch, move, recreate, or duplicate the raw dataset.

The Load Dataset process means scanning, validating, and registering metadata from the existing local directory. It does not mean downloading or copying the dataset.


The raw dataset directory is:

```text
backend/dataset/raw
```

All raw dataset data must be placed and read from:

```text
backend/dataset/raw
```

Use project-relative configuration where possible.

Reuse existing project conventions if equivalent directories already exist. Do not create duplicate storage structures without checking the current implementation first.

## Required CMS Audit Before Implementation

Inspect the current `/raw` CMS flow and create a short implementation audit before modifying it.

For every process below, classify the current implementation as:

```text
Existing
Partial
Missing
Broken
```

Audit these processes:

```text
Load Dataset
Dataset Profiling
Convert to YOLO-seg Format
Create data.yaml
Visualize Polygon Masks on Sample Images
Train YOLOv26m
Training Results and Curves
Validation Evaluation
Test Evaluation
Single Image Inference
Batch Inference on Test Set
Recycling Advice
ONNX Export
TorchScript Export
Final Verification
Per-Class Mask Metrics Report
```

For each item, verify:

- frontend page or component
- user action button
- frontend API call
- backend endpoint
- service or processing implementation
- generated artifact
- report data source
- error handling
- process status handling

Do not only provide an audit. Continue with implementation after the audit.

# CMS Navigation and Page Structure

Keep `/raw` valid.

The preferred behavior is:

```text
/raw
```

loads or redirects to:

```text
/raw/dataset
```

Create exactly these four sidebar menus:

```text
Dataset
Preparation
Training
Deployment
```

Each group below must be implemented as one sidebar menu and one page.

Inside each page:

- show one visual process datagrid
- each process is represented as a row or process section in that datagrid
- action buttons belong to the relevant process row
- report results appear in the same datagrid
- do not create a separate sidebar menu for each technical process
- do not require the user to move between pages to complete processes within the same group
- buttons must use simple labels suitable for non-technical users
- the datagrid must show actual execution status and actual generated report values
- clicking a process action must refresh or update its report row after execution
- long-running actions must show running status and progress without blocking the page

The required grouping is exactly:

```text
Dataset
  Load Dataset
  Dataset Profiling

Preparation
  Convert to YOLO-seg Format
  Visualize Polygon Masks on Sample Images

Training
  Train YOLOv26m
  Training Results and Curves
  Validation and Test Evaluation

Deployment
  Inference
  Batch Inference on Test Set and Recycling Advice
  Export Model
  Final Verification
```

# Dataset Menu

Route:

```text
/raw/dataset
```

This is one sidebar menu and one page.

Put these processes in the same visual datagrid:

```text
Load Dataset
Dataset Profiling
```

Do not create separate pages or sidebar items for these processes.

## Dataset Process Datagrid

Recommended columns:

```text
Process
Description
Status
Summary
Records
Duration
Last Run
Artifact
Action
```

The datagrid should initially contain process rows equivalent to:

| Process | Purpose | Action Button |
| --- | --- | --- |
| Load Dataset | Read and validate raw dataset | Load Dataset |
| Dataset Profiling | Analyze dataset structure, classes, image counts, and quality | Run Profiling |

The buttons must be placed in the `Action` column of the relevant process row.

Additional row actions may include:

```text
View Report
Download CSV
View Details
```

Do not use one large page-level button that hides which process is being executed.

## Load Dataset Requirements

Read from:

```text
backend/dataset/raw
```

The backend must:

- verify the dataset directory exists
- detect the main categories
- detect subcategories
- discover supported image files
- count images
- validate readable images
- detect unreadable images
- detect empty class directories
- detect unsupported files
- build the ordered class list
- build class ID mapping
- build main category and subcategory mapping
- build subcategory and main category mapping
- record process duration
- save a structured report

The local CMS must discover and report the actual local dataset structure instead of blindly hardcoding counts from the reference.

## Dataset Profiling Requirements

Generate actual profiling information:

- dataset path
- total image count
- main category count
- class count
- image count per category
- image count per class
- percentage per class
- minimum class size
- maximum class size
- imbalance ratio
- unreadable image count
- invalid file count
- empty directory count
- process duration
- process status
- run date

When the user clicks `View Report` for Dataset Profiling, show the detailed profiling report in the same page using an expandable row, drawer, modal, or detail panel. Do not add another sidebar menu.

# Preparation Menu

Route:

```text
/raw/preparation
```

This is one sidebar menu and one page.

Put these processes in the same visual datagrid:

```text
Convert to YOLO-seg Format
Visualize Polygon Masks on Sample Images
```

`data.yaml` generation is part of the conversion workflow. It must not become a separate sidebar menu.

## Preparation Process Datagrid

Recommended columns:

```text
Process
Description
Status
Progress
Output
Duration
Last Run
Artifact
Action
```

The datagrid should initially contain process rows equivalent to:

| Process | Purpose | Action Button |
| --- | --- | --- |
| Convert to YOLO-seg Format | Create train, validation, test data, polygon labels, and data.yaml | Convert Dataset |
| Visualize Polygon Masks on Sample Images | Generate sample overlays for polygon inspection | Generate Preview |

The buttons must be placed in the `Action` column of the relevant process row.

Additional row actions may include:

```text
View Report
View Preview
Download CSV
View Details
```

## Convert to YOLO-seg Format

Implement the conversion flow based on `walkthrought-reff.md`, adapted to the local application.

Required behavior:

- read original classification images
- preserve the discovered class mapping
- use reproducible random seed configuration
- create stratified train, validation, and test splits
- default to 70 percent train, 15 percent validation, and 15 percent test
- create YOLO segmentation image and label directories
- generate polygon masks
- try edge-based contour extraction first
- validate that an edge contour is meaningful before accepting it
- use a geometric polygon fallback when edge extraction fails
- normalize polygon coordinates
- write YOLO segmentation label format
- record whether each label used edge detection or fallback generation
- validate image and label pairing
- detect malformed labels
- detect invalid class IDs
- detect coordinates outside the normalized range
- generate `data.yaml`
- validate `data.yaml`
- save process reports and artifacts

Do not present generated masks as manually annotated ground truth. The report and documentation must clearly describe them as generated polygon masks or pseudo masks.

## data.yaml Requirements

Generate `data.yaml` as part of the `Convert to YOLO-seg Format` process.

It must contain:

```text
path
train
val
test
nc
names
```

Validate:

- file exists
- dataset path exists
- train path exists
- validation path exists
- test path exists
- class count matches the class mapping
- class names match generated label IDs

The conversion process detail report should show:

```text
Train Images
Train Labels
Validation Images
Validation Labels
Test Images
Test Labels
Edge Masks
Fallback Masks
Invalid Labels
Missing Pairs
data.yaml Status
```

## Visualize Polygon Masks on Sample Images

Generate sample mask overlays before training.

Each preview should include:

- original image
- polygon mask overlay
- polygon boundary
- class label
- split name
- mask generation strategy if available

Save generated previews as artifacts and make them viewable from the `View Preview` button in the same Preparation page.

Do not claim that masks are correct only because visualization generation succeeded. Report that the previews were generated for visual inspection.

# Training Menu

Route:

```text
/raw/training
```

This is one sidebar menu and one page.

Put these processes in the same visual datagrid:

```text
Train YOLOv26m
Training Results and Curves
Validation and Test Evaluation
```

Do not create separate sidebar menus for training, curves, validation, or test evaluation.

## Training Process Datagrid

Recommended columns:

```text
Process
Description
Status
Progress
Key Result
Duration
Last Run
Artifact
Action
```

The datagrid should initially contain process rows equivalent to:

| Process | Purpose | Action Button |
| --- | --- | --- |
| Train YOLOv26m | Train the waste segmentation model | Start Training |
| Training Results and Curves | Read and display actual training metrics and curve artifacts | View Results |
| Validation and Test Evaluation | Evaluate box and mask metrics on validation and test sets | Run Evaluation |

The buttons must be placed in the `Action` column of the relevant process row.

Additional row actions may include:

```text
View Report
View Curves
View Metrics
Download CSV
View Details
```

## Train YOLOv26m

Use the YOLOv26m segmentation model flow from `walkthrought-reff.md`, adapted to the actual local environment and installed library version.

Before starting training, allow the user to view a simple configuration summary:

```text
Model
Dataset
Image Size
Epochs
Batch Size
Patience
Device
Workers
Seed
Output Directory
```

Training must run as a background process or use the existing project job architecture.

The frontend request must not remain blocked for the complete training duration.

Expose:

```text
Not Started
Ready
Running
Completed
Failed
Blocked
```

Training progress should include:

- current epoch
- total epochs
- elapsed time
- latest loss values
- latest validation metrics
- best epoch when known
- best checkpoint path
- last checkpoint path
- failure reason

Important:

- verify actual optimizer selection from runtime logs
- do not report MuSGD if runtime selected another optimizer
- do not claim architecture features are active without runtime or model configuration evidence
- do not hide deprecation warnings
- do not fabricate completed epochs
- store actual training configuration
- store actual runtime environment

## Training Results and Curves

The `View Results` or `View Curves` action must open results within the same Training page.

Read actual artifacts from the training run.

Support available artifacts such as:

- results curve
- confusion matrix
- normalized confusion matrix
- precision curve
- recall curve
- F1 curve
- PR curve
- mask precision curve
- mask recall curve
- mask F1 curve
- mask PR curve

Display only artifacts that actually exist.

If an expected artifact is absent, show:

```text
Not Available
```

Do not create fake placeholder metric values.

## Validation and Test Evaluation

The `Run Evaluation` action should run validation and test evaluation as one grouped process.

Load the actual best checkpoint and evaluate both splits separately.

Collect box metrics:

```text
Precision
Recall
mAP50
mAP50-95
```

Collect mask metrics:

```text
Precision
Recall
mAP50
mAP50-95
```

Also collect per-class metrics when provided by the evaluation result.

The process row should show a concise summary. The `View Metrics` action should show detailed validation and test tables in the same Training page.

Keep validation and test results clearly distinguishable.

# Deployment Menu

Route:

```text
/raw/deployment
```

This is one sidebar menu and one page.

Put these processes in the same visual datagrid:

```text
Inference
Batch Inference on Test Set and Recycling Advice
Export Model
Final Verification
```

Do not create separate sidebar menus for inference, batch inference, export, or final verification.

## Deployment Process Datagrid

Recommended columns:

```text
Process
Description
Status
Summary
Duration
Last Run
Artifact
Action
```

The datagrid should initially contain process rows equivalent to:

| Process | Purpose | Action Button |
| --- | --- | --- |
| Inference | Upload one image and return segmentation masks, labels, and advice | Analyze Image |
| Batch Inference on Test Set and Recycling Advice | Analyze the test dataset and generate prediction and advice reports | Analyze Test Dataset |
| Export Model | Export the trained model to ONNX or TorchScript | Export Model |
| Final Verification | Verify artifacts and generate the per-class mask mAP report | Run Final Verification |

The buttons must be placed in the `Action` column of the relevant process row.

Additional row actions may include:

```text
View Result
View Report
Download CSV
Download Artifact
View Details
```

## Inference

When the user clicks `Analyze Image`, open an upload dialog, drawer, or inline form on the same Deployment page.

Required behavior:

- upload one supported image
- validate the upload
- run inference with the best model
- return segmentation masks
- return class labels
- return confidence scores
- return bounding box data
- return mask pixel count when available
- map subcategory to main waste category
- return recycling advice
- save the annotated output image
- save structured inference report data

The result view should show:

- uploaded image
- segmented output image
- detected class
- main category
- confidence
- recycling advice
- detection count
- mask information

## Batch Inference on Test Set and Recycling Advice

The `Analyze Test Dataset` button runs inference on the actual processed test split.

Generate report rows containing:

- input image
- expected class if derivable from dataset metadata
- predicted class
- main category
- confidence
- detection count
- mask availability
- recycling advice
- annotated output artifact
- duration
- status
- error if failed

The detailed batch result table must be accessible from the same Deployment page.

## Export Model

When the user clicks `Export Model`, open a simple format selector on the same Deployment page.

Supported formats:

```text
ONNX
TorchScript
```

Allow the user to select one format or both formats.

For each export, report:

- source checkpoint
- export format
- start time
- completion time
- duration
- output path
- file size
- export status
- validation status
- error message

Do not mark an export as successful only because the export command returned without an exception.

Verify:

- exported file exists
- exported file size is greater than zero
- load or runtime validation succeeds where supported

Package supporting metadata when appropriate:

- class names
- class ID mapping
- category mapping
- recycling advice mapping
- dataset configuration
- metric summary
- model configuration

## Final Verification

The `Run Final Verification` button must verify actual generated artifacts.

Check:

- raw dataset path exists
- raw dataset can be scanned
- processed dataset exists
- train image and label counts match
- validation image and label counts match
- test image and label counts match
- YOLO segmentation labels are valid
- `data.yaml` exists
- `data.yaml` paths are valid
- class count matches class mapping
- best checkpoint exists
- best checkpoint can be loaded
- validation metrics exist
- test metrics exist
- single-image inference succeeds
- segmentation masks are returned for a valid detected sample
- annotated inference output can be saved
- recycling advice mapping is available
- requested ONNX export exists and validates
- requested TorchScript export exists and validates
- per-class mask metrics are available
- final verification report is saved

The technical reference performs packaged model verification by loading the packaged best checkpoint and running prediction on a test image. Preserve this practical runtime verification behavior.

## Per-Class Mask mAP Report

The Final Verification process must expose a detailed per-class mask metrics table on the same Deployment page.

Required columns:

```text
Class
Images
Instances
Mask Precision
Mask Recall
Mask mAP50
Mask mAP50-95
Verification Status
```

The table must use actual evaluation output.

Do not copy example metric values from `walkthrought-reff.md`.

# Shared Visual Datagrid Behavior

There are exactly four main process datagrids:

```text
Dataset Process Datagrid
Preparation Process Datagrid
Training Process Datagrid
Deployment Process Datagrid
```

Each datagrid corresponds to one sidebar menu.

Each datagrid contains the processes assigned to that group.

Action buttons belong to the relevant process row.

The UI should follow this interaction pattern:

```text
User opens sidebar menu
Process datagrid is displayed
User clicks action button in a process row
Process status changes to Running
Backend process executes
Process row updates with result summary
User can open detailed report from the same page
User can download CSV report when available
```

Do not use arrow characters when implementing this flow in project files.

Create or reuse one configurable datagrid component instead of building unrelated table implementations.

Required features:

- process row actions
- pagination for detailed report data
- sorting
- search
- filters
- refresh
- CSV export
- status display
- run date
- duration
- artifact access
- loading state
- empty state
- error state
- responsive layout

Each process execution should produce structured report records with fields equivalent to:

```text
run_id
process_group
process_name
status
started_at
completed_at
duration_seconds
input_path
output_path
parameters
summary
metrics
artifacts
error_message
```

Use the current project persistence approach if one already exists. Do not introduce a new database or storage system without checking the existing architecture.

# Process Dependencies

Enforce dependencies in both backend validation and frontend button state.

Required dependency flow:

```text
Load Dataset
Dataset Profiling
Convert to YOLO-seg Format
Create data.yaml and Validate Labels
Visualize Polygon Masks
Train YOLOv26m
Training Results
Validation Evaluation
Test Evaluation
Inference
Batch Inference
Export Model
Final Verification
```

Do not use arrow characters in project files. The sequence above is represented as separate lines intentionally.

Examples:

- Dataset Profiling requires a readable dataset.
- Conversion requires a successful dataset scan and class mapping.
- Training requires a valid processed dataset and valid `data.yaml`.
- Evaluation requires a trained model.
- Inference requires a loadable model.
- Export requires a loadable model.
- Final Verification requires all mandatory artifacts for the selected workflow.

When a process is blocked, the CMS must explain the missing prerequisite.

# Legacy API Compatibility

The current project may already expose endpoints under `/api/kaggle/*`.

Audit all frontend consumers before renaming existing endpoints.

If `/api/kaggle/download?source=local` already exists and is required for compatibility, preserve it only as a legacy local-load action. It must not download, clone, fetch, move, or duplicate the dataset.

Prefer clear internal service naming such as dataset load, dataset scan, or dataset registration even when the public compatibility route remains unchanged.

# API and Backend Requirements

Inspect the existing backend architecture first.

Follow the current project conventions for:

- routers
- services
- schemas
- repositories
- jobs
- background tasks
- report storage
- configuration
- error handling
- logging

Do not place all processing logic directly inside route handlers.

The backend must expose enough API functionality for the CMS to:

- scan dataset
- profile dataset
- start conversion
- read conversion status
- validate generated labels
- generate mask previews
- start training
- read training status
- read training results
- run validation evaluation
- run test evaluation
- upload image for inference
- start batch inference
- export model
- run final verification
- list process reports
- retrieve report details
- export report CSV
- access generated artifacts safely

Use existing endpoint naming patterns if they already exist.

# Update walkthrought.md

After implementation, update `walkthrought.md`.

Keep it simple and aligned with the actual CMS.

Required sections:

```text
Project Overview
Dataset Location
CMS Workflow
Dataset
Preparation
Training
Deployment
Report Datagrids
Backend Processing
Frontend Routes
API Flow
Artifact Structure
Full Cycle Testing
Known Limitations
Final Verification
```

Do not paste notebook code into `walkthrought.md`.

For each CMS menu, explain only:

- what the user clicks
- what the backend executes
- what report is generated
- where artifacts are stored
- what prerequisite is required

# Reference Metrics Policy

Any metrics documented in `walkthrought-reff.md`, including best epoch results and per-class mask AP values, are reference results only.

The CMS must never present reference values as results from the current local execution.

All report tables, training summaries, validation metrics, test metrics, and per-class mask metrics must come from actual generated run artifacts.

If actual metrics are unavailable, show `Not Available`.

# Full Cycle Test Requirement

After implementation, run the actual frontend and backend test cycle.

Do not provide a theoretical test report.

Do not mark tests as passed unless they were executed.

Record:

- exact command
- exit code
- relevant output summary
- failure reason
- fix applied
- retest result

## Backend Test Scope

At minimum verify:

- backend dependency installation
- backend application startup
- health endpoint
- dataset path discovery
- dataset loading
- dataset profiling
- YOLO-seg conversion endpoint
- split generation
- polygon label generation
- label validation
- `data.yaml` generation
- mask preview generation
- training job creation
- training status endpoint
- training result retrieval
- validation evaluation endpoint
- test evaluation endpoint
- single-image inference endpoint
- batch inference endpoint
- recycling advice response
- ONNX export endpoint
- TorchScript export endpoint
- final verification endpoint
- report list endpoint
- report detail endpoint
- CSV export endpoint
- artifact access

## Training Test Strategy

Do not run the complete 120-epoch production training during routine full-cycle integration testing unless explicitly required and practical.

Use a smoke training configuration that verifies integration, for example:

```text
epochs: 1
small dataset fraction
small batch size
reduced workers if needed
```

The smoke test validates:

- job creation
- dataset loading
- model loading
- one training cycle
- checkpoint generation
- metric generation
- result parsing
- frontend progress handling

Clearly label smoke-training metrics as smoke-test results, not final research results.

## Frontend Test Scope

At minimum verify:

- `/raw` remains valid
- Dataset menu navigation
- Preparation menu navigation
- Training menu navigation
- Deployment menu navigation
- primary action buttons
- secondary action buttons
- prerequisite blocking
- loading state
- running state
- completed state
- failed state
- datagrid rendering
- datagrid filtering
- datagrid sorting
- datagrid pagination
- CSV export
- artifact access
- image upload
- segmented output preview
- recycling advice display
- training progress display
- training curve display
- validation metric display
- test metric display
- per-class mask metric table
- model export controls
- final verification result display

Use the project's existing frontend test framework and E2E framework if available.

## Integration Smoke Flow

Execute the complete smoke flow in dependency order:

```text
Load Dataset
Dataset Profiling
Convert Dataset
Validate Labels
Generate data.yaml
Generate Mask Preview
Start Smoke Training
Read Training Results
Run Validation
Run Test Evaluation
Run Single Image Inference
Run Batch Inference
Export ONNX
Export TorchScript
Run Final Verification
Generate CSV Reports
```

If a step fails:

- record the failure
- identify the cause
- fix it when within project scope
- rerun the failed step
- rerun dependent steps if needed
- report the final actual status

# Required Test Report

Create:

```text
docs/full-cycle-test-report.md
```

Required sections:

```text
Environment
Test Date
Frontend Version
Backend Version
Python Version
Node Version
GPU or CPU Device
Dataset Path
Dataset Summary
Commands Executed
Backend Test Results
Frontend Test Results
Integration Test Results
Smoke Training Result
Validation Result
Test Evaluation Result
Inference Result
Batch Inference Result
Export Result
Final Verification Result
Report Export Result
Known Failures
Warnings
Conclusion
```

Use a result table with these columns:

```text
Test Area
Test Case
Command or Endpoint
Expected Result
Actual Result
Status
Notes
```

Allowed status values:

```text
Passed
Failed
Blocked
Skipped
```

Every `Skipped` or `Blocked` result must include a reason.

# Final Delivery Requirements

At the end of the implementation, provide:

- CMS audit before changes
- files changed
- frontend routes added or updated
- backend endpoints added or updated
- service modules added or updated
- report datagrids added
- dataset artifact locations
- training artifact locations
- inference artifact locations
- export artifact locations
- test commands executed
- test result summary
- remaining failures or limitations
- location of `docs/full-cycle-test-report.md`

The implementation is complete only when the CMS flow, backend processing, report datagrids, artifact generation, and actual full-cycle test report are aligned with the implemented code.