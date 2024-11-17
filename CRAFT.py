#pip install craft-text-detector
def CRAFT(image_dir,output_dir):
  """

  Args:
    image_dir: The directory of original images
    output_dir: The directory of output images
  """
  # import
  from craft_text_detector import (read_image,load_craftnet_model,load_refinenet_model,get_prediction,export_detected_regions,export_extra_results,empty_cuda_cache)

  import os
  import torch

  CUDA=True if torch.cuda.is_available else 'cpu'

  #get the path of images
  def read(path):
      myPath = path
      otherList=os.walk(myPath)
      PATH = []
      for root, dirs, files in otherList:
        if root!=myPath:
        for i in files:
        PATH.append(root+str("/")+str(i))

  images = read(image_dir)

  #set the output directory
  output_dir = output_dir

  #read the model
  refine_net = load_refinenet_model(cuda=CUDA)
  craft_net = load_craftnet_model(cuda=CUDA)


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
  if CUDA:
    empty_cuda_cache()