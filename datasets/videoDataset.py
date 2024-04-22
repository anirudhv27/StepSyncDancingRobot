import torch
import torch
from torch.utils.data import Dataset, DataLoader

class TrajectoryDataset(Dataset):
    '''
    Takes in a list of dictionaries as a trajectory, converts into a sequence of feature vectors represe    nting frames.
    
    Each dictionary has the key 'angles' and '    positions', which each have a list. Each list element in angles is a 4-element quaternion, 
    and each element in positions is a 3-element position. 
    
    Convert each dictionary into a feature vector with 8 elements, where the first 4 elements are the quaternion angles, next 3 elements are the positions,
    and last element is the normalized index (from 0 to 1).
    '''
    
    def __init__(self, trajectory, num_frames_per_example=4):
        self.feature_vectors = []
        self.num_frames_per_example = num_frames_per_example
        
        for frame_idx, frame in enumerate(trajectory):
            angles = frame['angles']
            positions = frame['positions']
            
            num_angles = len(angles)
            num_positions = len(positions)
            feature_vector_len = 4*num_angles + 3*num_positions + 1
            feature_vector = torch.zeros(feature_vector_len);
            
            for i, joint in enumerate(angles):
                feature_vector[i:i+4] = torch.tensor(angles[joint])
                
            for i, point in enumerate(positions):
                feature_vector[num_angles*4 + i:num_angles*4 + i+3] = torch.tensor(positions[point])
            
            feature_vector[-1] = torch.tensor(frame_idx / (num_angles - 1))
            self.feature_vectors.append(feature_vector)

    def __len__(self):
        return len(self.feature_vectors) - len(self.num_frames_per_example) + 1

    def __getitem__(self, index):
        vector_list = self.feature_vectors[index:index+self.num_frames_per_example]
        return torch.stack(vector_list, dim=0)