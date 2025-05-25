import numpy as np
from sklearn.model_selection import train_test_split

def normalize_array_length(arr1, arr2):

    if len(arr1) < len(arr2):
        cut_amt = len(arr2) - len(arr1)
        return arr1, arr2[cut_amt:]
    elif len(arr1) > len(arr2):
        cut_amt = len(arr1) - len(arr2)
        return arr1[cut_amt:], arr2    
    else:
        return arr1, arr2

def train_val_history_idx_split(Xfull, history, train_size=0.8, shuffle=True):
	
	val_size = 1 - train_size
	all_idxs = np.arange(history, len(Xfull)-1)
	train_idxs, val_idxs, _, _ = train_test_split(all_idxs, all_idxs, test_size=val_size, random_state=42, shuffle=shuffle)	
	return train_idxs, val_idxs

# Create a sliding window of time series data,
def transform_to_window_data(dataset, target, history, target_size=1):
		data = []
		targets = []

		start_index = history
		end_index = len(dataset) - target_size

		for i in range(start_index, end_index):
			indices = range(i - history, i)
			data.append(dataset[indices])
			targets.append(target[i+target_size])

		return np.array(data), np.array(targets)

# Generic data generator object for feeding data
def reconstruction_errors_by_idxs(event_detector, Xfull, idxs, history, bs=4096):
    
    # Length of reconstruction errors is len(X) - history. Clipped from the front.
    full_errors = np.zeros((len(idxs), Xfull.shape[1]))
    idx = 0

    for idx in range(0, len(idxs), bs):
        
        Xbatch = []
        Ybatch = []

        # Build the history out by sampling from the list of idxs
        for b in range(bs):
            
            if idx + b >= len(idxs):
                break
            
            lead_idx = idxs[idx+b]
            Xbatch.append(Xfull[lead_idx-history:lead_idx])
            Ybatch.append(Xfull[lead_idx+1])

        Xbatch = np.array(Xbatch)
        Ybatch = np.array(Ybatch)

        if idx + bs > len(full_errors):
            full_errors[idx:] = (event_detector.predict(Xbatch) - Ybatch)**2                
        else:
            full_errors[idx:idx+bs] = (event_detector.predict(Xbatch) - Ybatch)**2

    return full_errors

def custom_train_test_split(dataset_name, Xtest, Ytest, test_size=0.7, shuffle=False):

    if dataset_name == 'BATADAL':
        # The first 30% of BATADAL contains no attacks, so we use the back 30% instead
        Xtest_test, Xtest_val, Ytest_test, Ytest_val = train_test_split(Xtest, Ytest, test_size=1-test_size, shuffle=shuffle)
    else:
        Xtest_val, Xtest_test, Ytest_val, Ytest_test = train_test_split(Xtest, Ytest, test_size=test_size, shuffle=shuffle)

    return Xtest_val, Xtest_test, Ytest_val, Ytest_test