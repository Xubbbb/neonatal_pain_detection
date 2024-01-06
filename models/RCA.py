import torch
import torch.nn as nn
import torchvision.models as models
from .CustomVGG16FeatureExtractor import CustomVGG16FeatureExtractor

# Region-Channel-Attention Module
class RCA(nn.Module):
    def __init__(self, num_classes = 8, num_frames = 20):
        super(RCA, self).__init__()
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.feature_extractor = CustomVGG16FeatureExtractor()
        self.flatten = nn.Flatten()
        self.local_fc_layers = nn.ModuleList([nn.Linear(512*7*7, 512) for _ in range(6)])
        self.local_sigmoid_layers = nn.ModuleList([nn.Sigmoid() for _ in range(6)])
        self.global_fc_layers = nn.ModuleList([nn.Linear(1024*7*7, 512) for _ in range(6)])
        self.global_sigmoid_layers = nn.ModuleList([nn.Sigmoid() for _ in range(6)])
        self.final_fc_layer = nn.Linear(1024*7*7, self.num_classes)
        # Fake output
        self.final_time_fc_layer = nn.Linear(self.num_frames * self.num_classes, self.num_classes)
        
    def forward(self, x):
        frame_outputs = []
        for t in range(x.size(1)):
            # Fi list
            local_feature_list = []
            # Corresponding to the paper's μ
            local_weights_list = []
            
            # 1.Local self-attention stage
            for i in range(6):
                # Select a region
                img = x[:, t, i, :, :, :]
                
                # For pretrained model, we don't need to calculate gradients
                with torch.no_grad():
                    local_feature = self.feature_extractor(img)
                local_feature_list.append(local_feature)
                feature_weight = self.flatten(local_feature)
                feature_weight = self.local_fc_layers[i](feature_weight)
                feature_weight = self.local_sigmoid_layers[i](feature_weight)
                local_weights_list.append(feature_weight)
            
            # Compact representation in the jth channel(based on every region's jth channel's attention weight)
            
            # Initialize FG, the aggregated feature map
            aggregated_feature_map = torch.zeros_like(local_feature_list[0])
            # Normalize the weights using linear normalization
            normalized_weights_list = [weight / weight.sum(dim=1, keepdim=True) for weight in local_weights_list]
            
            
            # Aggregate the feature maps based on weights
            for channel in range(512):
                for i in range(6):
                    # Apply attention weight to each feature map
                    weighted_feature_map = local_feature_list[i][:, channel, :, :] * normalized_weights_list[i][:, channel].view(-1, 1, 1)
                    aggregated_feature_map[:, channel, :, :] += weighted_feature_map
            
            # Fi | FG list
            global_feature_list = [torch.cat((local_feature_map, aggregated_feature_map), dim=1) for local_feature_map in local_feature_list]
            
            # Then, we concatenate local features and compact representation FG as input 
            # to another FC layer to calculate new attention weights. 
            refined_weights_list = []
            
            # 2.Global refinement stage
            for i in range(6):
                gloabl_feature = self.flatten(global_feature_list[i])
                global_weight = self.global_fc_layers[i](gloabl_feature)
                global_weight = self.global_sigmoid_layers[i](global_weight)
                refined_weights_list.append(global_weight)
                
            # Subsequently, on the basis of the refined attention weight, 
            # we summarized all the regional features into a refined compact facial representation α  
            final_weights_list = []
            for local_weight, refined_weight in zip(local_weights_list, refined_weights_list):
                multiplied_weight = local_weight * refined_weight
                normalized_weight = multiplied_weight / multiplied_weight.sum(dim=1, keepdim=True)
                final_weights_list.append(normalized_weight)
                
            refined_compact_representation = torch.zeros_like(aggregated_feature_map)
            
            for channel in range(512):
                for i in range(6):
                    weighted_feature_map = local_feature_list[i][:, channel, :, :] * final_weights_list[i][:, channel].view(-1, 1, 1)
                    refined_compact_representation[:, channel, :, :] += weighted_feature_map
                    
            # Finally, we feed the two compact features (α and FG ) into a fully-connected layer Φ2 
            # separately to obtain their respective feature vectors ∈ R1×L×1 
            # (the dimension of tensor represent the temporal, spatial, and channel sizes accordingly), 
            # and the two feature vectors are concatenated as the final output α ̃.
            
            #! To be used in future work, the output size will be num_classes * 2
            # refined_compact_representation = self.flatten(refined_compact_representation)
            # refined_compact_representation = self.final_fc_layer(refined_compact_representation)
            # aggregated_feature_map = self.flatten(aggregated_feature_map)
            # aggregated_feature_map = self.final_fc_layer(aggregated_feature_map)
            # output = torch.cat((refined_compact_representation, aggregated_feature_map), dim=1)
            
            #! Now, we only directly FC two feature vectors into an output size of num_classes
            final_compact_representation = torch.cat((refined_compact_representation, aggregated_feature_map), dim=1)
            final_compact_representation = self.flatten(final_compact_representation)
            frame_output = self.final_fc_layer(final_compact_representation)
            frame_outputs.append(frame_output)
            
        concatenated_frame_outputs = torch.cat(frame_outputs, dim=1)
        final_output = self.final_time_fc_layer(concatenated_frame_outputs)
        
        return final_output
        
        
        
