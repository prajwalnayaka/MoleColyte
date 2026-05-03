import torch
data=torch.load("Preprocessed Datasets/tox21_3d_fgn_dataset.pt", weights_only=False)
print(data[37])
#print(data[7].edge_index)
#print(data[4367].edge_attr)
#print(data[7].pos)
#print(data[65].y)

# import torch
#
# # Load the dataset
# data = torch.load("tox21_3d_fgn_dataset.pt", weights_only=False)
#
# virtual_bond_count = 0
# graphs_with_fgns = 0
#
# # Loop through every graph and check the 5th column (index 4)
# for i, graph in enumerate(data):
#     # Check if the 5th column has any 1s
#     num_virtual = (graph.edge_attr[:, 4] == 1).sum().item()
#
#     if num_virtual > 0:
#         virtual_bond_count += num_virtual
#         graphs_with_fgns += 1
#
# print(f"Total Virtual Bonds injected: {virtual_bond_count}")
# print(f"Molecules with FGNs: {graphs_with_fgns} / {len(data)}")