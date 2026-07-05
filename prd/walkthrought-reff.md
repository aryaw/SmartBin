2. Explore Original Dataset Structure
3. Convert to YOLO-seg Format (Polygon Masks)
4. Create data.yaml
5. Visualize Polygon Masks on Images
6. Train YOLOv26m-seg with Optimized Hyperparameters
7. Training Results & Curves
8. Validation & Test Evaluation (Box + Mask Metrics)
9. 🎯 Inference — Output Images with Segmentation Masks & Labels
10. Run on Original Dataset Images + Recycling Advice
11. Export & Package for Deployment
12. Final Verification


- dont download data its already under dir Waste_Classification_Dataset
- does all this flow already in cms?
what better button and sidebar menu flow we put under http://localhost:3000/raw to follow those process
- all data from raw under dir Waste_Classification_Dataset will be put in directory 
backend/dataset
- use walkthrought.md as reffrence make it simple
- also create datagrid report for each process, as paper to report to Prof.
- dont use em dash, or →, also dont use emoji for all in project directory
- create datagrid and sidebar menu for each proces so there's will be report for each proces
- use simpllify menu for user non tech, gropu menu by (each group represent 1 sidebar menu, dont put any number just use menu):
 put this in same data grid to report
 * Load Dataset (Waste_Classification_Dataset)
 * Dataset Profiling

 put this in same data grid to report:
 * Convert to YOLO-seg Format
 * Visualize Polygon Masks on Sample Images

 put this in same data grid to report:
 * Train YOLOv26m
 * Training Results & Curves
 *  Validation & Test Evaluation (Box + Mask Metrics)

 put this in same data grid to report:
 * Inference - (Upload Image, Segmentation Masks and Advice)
 * Batch Inference on Test Set and Recycling Advice
 * Export Model (ONNX / TorchScript)
 * Final Verification: per-class mask mAP table report

- run full cycle test frontend and backend, give full report from the test
