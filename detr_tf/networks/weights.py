import os
import requests


WEIGHT_NAME_TO_CKPT = {
    "detr": [
        "https://storage.googleapis.com/visualbehavior-publicweights/detr/checkpoint",
        "https://storage.googleapis.com/visualbehavior-publicweights/detr/detr.ckpt.data-00000-of-00001",
        "https://storage.googleapis.com/visualbehavior-publicweights/detr/detr.ckpt.index"
    ]
}

def load_weights(model, weights: str):
    """ Load weight on a given model
    Weights are supposed to be stored in the weight folder at the root of the repository. If weights
    do not exist, but are publicly known, the weight will be downloaded from gcloud.
    """
    if not os.path.exists('weights'):
        os.makedirs('weights')
    
    if weights.endswith('.ckpt'):
        # Convert .ckpt to .h5 or .keras format if needed
        ckpt_path = os.path.join('weights', weights)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        for f in WEIGHT_NAME_TO_CKPT[weights]:
            fname = f.split("/")[-1]
            if not os.path.exists(os.path.join(ckpt_path, fname)):
                print("Downloading...", f)
                r = requests.get(f, allow_redirects=True)
                open(os.path.join(ckpt_path, fname), 'wb').write(r.content)
        print("Restoring weights from", os.path.join(ckpt_path, f"{weights}"))
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(os.path.join(ckpt_path, f"{weights}")).expect_partial()
    elif weights.endswith('.h5') or weights.endswith('.keras'):
        # Directly load .h5 or .keras files
        weight_path = os.path.join('weights', weights)
        if not os.path.exists(weight_path):
            print("Downloading...", weights)
            r = requests.get(weight_path, allow_redirects=True)
            open(weight_path, 'wb').write(r.content)
        print("Loading weights from", weight_path)
        model.load_weights(weight_path)
    else:
        raise Exception(f'Cannot load the weights: {weights}')
