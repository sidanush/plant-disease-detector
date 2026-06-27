import numpy as np

classes = [
    'Apple___Black_rot', 'Apple___Healthy', 'Apple___Scab',
    'Potato___Black_leg', 'Potato___Early_blight', 'Potato___Healthy',
    'Potato___Late_blight', 'Tomato___Early_blight', 'Tomato___Healthy',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold'
]

np.save("classes.npy", classes)
print("✅ classes.npy saved!")
