import time
import multiprocessing as mp
from PIL import Image
import pandas as pd

HEIGHT = 137
WIDTH = 236

def get_images_and_labels(df, height=HEIGHT, width=WIDTH):
    return df.iloc[:,0], df.iloc[:,1:].values.reshape(-1, height, width)

def preprocess_img(img):
    # Images are inverted
    img = 255 - img
    return img

def worker_fn(queue, filename, images, save_dir, start_time):
    while True:
        try:
            i = queue.get()
            
            if i is None:
                break
            
            fn = filename.iloc[i]
            img = images[i]
            img = preprocess_img(img)
            img = Image.fromarray(img).convert('RGB')
            img.save('{}.png'.format(save_dir/fn))
            if i % 5000 == 0:
                print('Completed {:4f}% in {:4f}'.format(
                    i / len(images) * 100,
                    time.time() - start_time
                ))
            
        except e:
            print(e)
            break

def gen_preprocessed_data(dataset_paths, save_dir):
    start_time = time.time()

    for data_path in dataset_paths:
        df_start_time = time.time()

        df = pd.read_parquet(data_path)
        filename, images = get_images_and_labels(df)
        assert len(filename) == len(images)
        
        n_workers = 10
        queue = mp.Queue()
        workers = [mp.Process(target=worker_fn,
                              args=(queue, filename, images, save_dir, df_start_time))
                      for i in range(n_workers)]
        # Add indices
        for i in range(len(images)):
            queue.put(i)
        # Add Nones to end of queue; ends workers
        for i in range(n_workers):
            queue.put(None)
        
        # Start workers
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        
        print('Total time for df: ', time.time() - df_start_time)
        print()

    print('Total time: ', time.time() - start_time)