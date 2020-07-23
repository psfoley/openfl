import pandas as pd
import numpy as np
from tfedlrn import TensorKey

class TensorDB(object):
    """
    The TensorDB stores a tensor key and the data that it corresponds to. It is built on top of a pandas dataframe
    for it's easy insertion, retreival and aggregation capabilities. Each collaborator and aggregator has its own TensorDB.
    """
    def __init__(self):
        self.tensor_db = pd.DataFrame([], columns=['tensor_name','origin','round','tags','nparray'])

    def cache_tensor(self, tensor_key_dict):
        """
        Insert tensor into TensorDB (dataframe)

	Parameters:
	-----------
        tensor_key_dict:	{tensor_key: nparray}

        Returns
	-------
	None
        """

        for tensor_key,nparray in tensor_key_dict.items():
            tensor_name = tensor_key[0]
            origin = tensor_key[1]
            round_num = tensor_key[2]
            tags = tensor_key[3]
            df = pd.DataFrame([tensor_name,origin,round,tags,nparray], \
                              columns=['tensor_name','origin','round','tags','nparray'])
            self.tensor_db = pd.concat([self.tensor_db,df],ignore_index=True)


    def get_tensor_from_cache(self, tensor_key):
        """
        Performs a lookup of the tensor_key in the TensorDB. Returns the nparray if it is available
        Otherwise, it returns 'None'
        """

        #TODO come up with easy way to ignore compression

        df = self.tensor_db[(self.tensor_db['tensor_name'] == tensor_key[0]) & \
                            (self.tensor_db['origin'] == tensor_key[1]) & \
                            (self.tensor_db['round_num'] == tensor_key[2]) & \
                            (self.tensor_db['tags'] == tensor_key[3])]
        if len(df) == 0:
            return None
        return df['nparray'][0]  

    def get_aggregated_tensor(self, tensor_key, collaborator_names, collaborator_weights):
        """
        Determines whether all of the collaborator tensors are present for a given tensor key, and returns their weighted average 

        Parameters
        ----------
        tensor_key:		The tensor key to be resolved. If origin 'agg_uuid' is present, 
                                can be returned directly. Otherwise must compute weighted average of all collaborators
        collaborator_names:	List of collaborator names in federation
        collaborator_weights:	{col_name: weight}

        Returns
        -------
        weighted_nparray if all collaborator values are present
        None if not all values are present
        
        """
        assert(sum(collaborator_weights.values()) == 1.0), "Collaborator weights are not normalized"
        agg_tensor_dict = {}
        for col in collaborator_names:
             new_tags = tuple(list(tensor_key[3]) + [col])
             agg_tensor_dict[col] = self.tensor_db[(self.tensor_db['tensor_name'] == tensor_key[0]) & \
                                                   (self.tensor_db['origin'] == tensor_key[1]) & \
                                                   (self.tensor_db['round_num'] == tensor_key[2]) & \
                                                   (self.tensor_db['tags'] == new_tags)]['nparray'][0]
             if len(agg_tensor_dict[col]) == 0:
                 return None
             agg_tensor_dict[col] = agg_tensor_dict[col] * collaborator_weights[col] 
        agg_nparray = np.sum([agg_tensor_dict[col] for col in collaborator_names],axis=0)

        return agg_nparray            
