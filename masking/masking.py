import torch
from google.cloud import storage
import tokenizers
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler
import numpy as np
import random
import jieba
import logging
logging.getLogger("jieba").setLevel(logging.WARNING)


tokenizer = BertWordPieceTokenizer(vocab_file = '../tokenizer/vocab.txt')
tokenizer.add_special_tokens(["<nl>"])
tokenizer.enable_truncation(max_length=512)
tokenizer.enable_padding(length=512)
client = storage.Client()
blobs = []
size = 0
for blob in client.list_blobs('tfrc-tfrc', prefix='public_model/corpus/'):
    if(blob.name.endswith('.txt')):
        blobs.append(blob)
        
sub_blobs = random.sample(blobs, 5)

def iterator_gen(generator, handler=None, parallel = False):
    try:
        import gc
        import multiprocessing as multiprocessing
        if parallel:
            cpu_count = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(cpu_count)

        if handler is not None:
            

            for e in (pool.imap(handler,generator) if parallel else iter(handler(e) for e in generator)):
                if e:
                    if isinstance(e, list):
                        for e in e:
                            yield e
                    else:
                        yield e
                    
        else:
            for e in generator:
                if e:
                    if isinstance(e, list):
                        for e in e:
                            yield e
                    else:
                        yield e
    finally:
        if parallel:
            try:
                pool.terminate()
            except:
                pass
            gc.collect()

def mask_ids(encoding, words):
    mask_no = int(round((len(encoding.ids) - (np.array(encoding.ids)<=5).sum()).item()*0.15))
    words_nospace = [word for word in words if word[0]!=' ']
    if(mask_no > len(words_nospace)):
        return None
    sample = random.sample(words_nospace, k=mask_no)
    masked_ids = np.array(encoding.ids)
    masked_words = 0
    for word in sample:
        start_index = [ind for ind, i in enumerate(encoding.offsets) if i[0] == word[1]]
        end_index = [ind for ind, i in enumerate(encoding.offsets) if i[1] == word[2]]
        if len(start_index)==0 or len(end_index)==0:
            continue
        else:
            start_index, end_index = start_index[0], end_index[0]
            if start_index == 0:
                start_index += 1
            masked_words += end_index - start_index + 1
            if random.random()<=0.1:
                masked_ids[start_index:end_index+1] = np.random.randint(6,50000,size = end_index - start_index + 1)
            elif random.random() <=0.8:
                masked_ids[start_index:end_index+1] = 4
        if masked_words > mask_no:
            break
    return masked_ids

#generator ver
import time

start = time.time()

total_size = 0
with open("/mnt/d/data_original", "wb") as f, open("/mnt/d/data_masked", "wb") as m:
    for count, blob in enumerate(sub_blobs):
        data = blob.download_as_string()
        data = data.decode("utf-8")
        data = data.split("\n\n")
        flat_data = []
        for line in data:
            if len(line) > 100000:
                line = [line[i:i+100000] for i in range(0, len(line), 100000)]
                flat_data.extend(line)
            else:
                flat_data.append(line)
        data = flat_data
        print(f"start tokenizing file {blob.name}")
        encoded = tokenizer.encode_batch(data)
        print(f"finish tokenizing file {blob.name}")
        # Prepare something for worker to do
        def generator():
            index = 0
            for item in encoded:
                yield(item, index)
                index += 1

        # Actual Work
        def worker(item):
            size = 0
            ids, masked_ids = [], []
            index = item[1]
            item = item[0]

            min_index = item.offsets[1][0] #get original index of first word in encoded segment
            max_index = max(item.offsets)[1] #get original index of last word in encoded segment
            words = list(jieba.tokenize(data[index][min_index:max_index]))
            arr = np.array(item.ids, dtype=np.int32)
            if(np.count_nonzero(arr) > 10):
                masked_id = mask_ids(item, words)
                if masked_id is not None:
                    ids.append(arr)
                    masked_ids.append(np.array(masked_id, dtype=np.int32))
                size += 1
                
            for overflowing in item.overflowing:
                min_index = overflowing.offsets[1][0]
                max_index = max(overflowing.offsets)[1]
                words = list(jieba.tokenize(data[index][min_index:max_index]))
                arr = np.array(overflowing.ids, dtype=np.int32)
                if(np.count_nonzero(arr) > 10):
                    masked_id = mask_ids(overflowing, words)
                    if masked_id is not None:
                        ids.append(arr)
                        masked_ids.append(np.array(masked_id, dtype=np.int32))
                    size += 1
                    
            return ids, masked_ids, size
            

        g = generator()
        print(f"start masking file {blob.name}")
        for ids, masked_ids, size in iterator_gen(g, worker, parallel=True):
            for i in masked_ids:
                m.write(i)
            for i in ids:
                f.write(i)
            total_size += size
        print(f"finish masking file {blob.name} - total size: {total_size}")

end = time.time()
print(end - start, " seconds")