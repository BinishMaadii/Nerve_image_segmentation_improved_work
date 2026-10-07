import os
from IPython.display import Image, display
plot_dir = "/kaggle/working/outputs/plots"

for file in os.listdir(plot_dir):
  if file.endswith((".png", ".jpeg", ".jpeg")):
    file_path = os.path.join(plot_dir, file)
    print(f"Display: {file}")
    display(Image(filename=file_path))
    
