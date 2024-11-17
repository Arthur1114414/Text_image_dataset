#pip install craft-text-detector

# import
from craft_text_detector import (read_image,load_craftnet_model,load_refinenet_model,get_prediction,export_detected_regions,export_extra_results,empty_cuda_cache)

import os

#get the path of images
def read(path):
    myPath = path
    otherList=os.walk(myPath)
    PATH = []
    for root, dirs, files in otherList:
       if root!=myPath:
       for i in files:
       PATH.append(root+str("/")+str(i))

images = read(image_path)

#set the output directory
output_dir = 'output/'

#read the model
refine_net = load_refinenet_model(cuda=True)
craft_net = load_craftnet_model(cuda=True)


for image in images:
	image = read_image(image)
	
	#predict
	pred_result = get_prediction(
    		image=image,
    		craft_net=craft_net,
    		refine_net=refine_net,
    		text_threshold=0.7, 
    		link_threshold=0.4, 
    		low_text=0.4,
    		cuda=True,
    		long_size=1280
		)

	# crop
	exported_file_paths = export_detected_regions(
    		image=image,
    		regions=pred_result["boxes"],
    		output_dir=output_dir,
    		rectify=True
		)

	# save
	export_extra_results(
	    image=image,
	    regions=pred_result["boxes"],
	    heatmaps=pred_result["heatmaps"],
	    output_dir=output_dir
		)

# off the GPU
empty_cuda_cache()
