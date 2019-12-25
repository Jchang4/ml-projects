from fastai.vision import *

def lr_find(learn):
    learn.lr_find()
    learn.recorder.plot()

def interp_learner(learn):
    interp = ClassificationInterpretation.from_learner(learn)
    losses,idxs = interp.top_losses()
    interp.plot_top_losses(9, figsize=(15,11))
    learn.recorder.plot_losses()
    print(interp.most_confused(min_val=2)[:10])

def create_data_bunch(df, path, folder, label_col, 
                      valid_pct=0.3, size=224, bs=256):
    tfms = get_transforms(do_flip=False, 
                          max_zoom=1.)
    return (ImageList
            .from_df(df, 
                     path/folder, 
                     cols='image_id', 
                     suffix='.png')
            .split_by_rand_pct(valid_pct=valid_pct, seed=42) # Remove seed to get true random split
            .label_from_df(cols=label_col)
            .transform(tfms, size=size, resize_method=ResizeMethod.SQUISH)
            .add_test(ImageList.from_df(pd.read_csv(path/'test.csv'),
                                        path/'test',
                                        cols='image_id',
                                        suffix='.png'))
            .databunch(bs=bs)
            .normalize(imagenet_stats))