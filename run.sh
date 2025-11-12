## HLN
# run
#python main.py --file_fold './Data/HLN/' --data_type '10x' --n_clusters 10 --init_k 10 --KNN_k 20 --RNA_weight 0.95 --ADT_weight 2.4 --tau 1.5 --vis_out_path 'results/HLN.png' --txt_out_path 'results/HLN.txt'
# eval
#python cal_matrics.py --GT_path './Data/HLN/GT_labels.txt' --our_path './results/HLN.txt' --save_path './results/HLN_metrics.txt'

## Mouse Brain
#python main.py --file_fold './Data/Mouse_Brain/' --data_type 'Spatial-epigenome-transcriptome' --n_clusters 14 --init_k 14 --KNN_k 20 --RNA_weight 1 --ADT_weight 7 --vis_out_path 'results/MB.png' --txt_out_path 'results/MB.txt'
# eval
#python cal_matrics.py --GT_path './Data/Mouse_Brain/MB_cluster.txt' --our_path './results/MB.txt' --save_path './results/MB_metrics.txt'

## Simulation
python main.py --file_fold './Data/Simulation/' --data_type 'Simulation' --n_clusters 5 --init_k 5 --KNN_k 20 --cl_weight 10 --vis_out_path 'results/Simulation.png' --txt_out_path 'results/Simulation.txt'
# eval
python cal_matrics.py --GT_path './Data/Simulation/GT.txt' --our_path './results/Simulation.txt' --save_path './results/Simulation_metrics.txt'



