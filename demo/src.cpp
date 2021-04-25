#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>
#include <string>

using namespace InferenceEngine;

static InferenceEngine::Blob::Ptr wrapMat2Blob( const cv::Mat &mat ) {
	size_t channels = mat.channels();
	size_t height = mat.size().height;
	size_t width = mat.size().width;
  
  size_t strideH = mat.step.buf[0];
  size_t strideW = mat.step.buf[1];
  
  bool is_dense = (strideW == channels && strideH == channels * width);
  if( !is_dense ) THROW_IE_EXCEPTION << "Doesn't support conversion from not dense cv::Mat";
 
 InferenceEngine::TensorDesc tDesc( InferenceEngine::Precision::U8, { 1, channels, height, width }, InferenceEngine::Layout::NCHW );
 return InferenceEngine::make_shared_blob<uint8_t>( tDesc, mat.data );
 }

int main(int argc, char** argv) 
{
	const std::string input_image_path{"../picture.jpg"};
	const std::string input_model{"../face-detection-adas-0001.xml"};
	
	// --------------------------- 1. Load inference engine instance -------------------------------------
	InferenceEngine::Core ie;
	// -----------------------------------------------------------------------------------------------------

    // 2. Read a model in OpenVINO Intermediate Representation (.xml and .bin files) format
	CNNNetwork network = ie.ReadNetwork(input_model);
    // -----------------------------------------------------------------------------------------------------
	
	// --------------------------- 3. Configure input & output ---------------------------------------------
    // --------------------------- Prepare input blobs -----------------------------------------------------
    InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    std::string input_name = network.getInputsInfo().begin()->first;

    input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
    input_info->setLayout(Layout::NCHW);
    input_info->setPrecision(Precision::U8);
    // --------------------------- Prepare output blobs ----------------------------------------------------
    DataPtr output_info = network.getOutputsInfo().begin()->second;
    std::string output_name = network.getOutputsInfo().begin()->first;

    const SizeVector outputDims = output_info->getTensorDesc().getDims();
    const int numPred = outputDims[2];
    std::vector<size_t> imageWidths, imageHeights;
	output_info->setPrecision(Precision::FP32);
    // -----------------------------------------------------------------------------------------------------
	
	// --------------------------- 4. Loading model to the device ------------------------------------------
    ExecutableNetwork executable_network = ie.LoadNetwork(network, "ARM");;
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 5. Create infer request -------------------------------------------------
    InferRequest infer_request = executable_network.CreateInferRequest();
    // -----------------------------------------------------------------------------------------------------
	
	// --------------------------- 6. Prepare input --------------------------------------------------------
    cv::Mat image = cv::imread(input_image_path);
    int w = image.cols;
    int h = image.rows;
    Blob::Ptr imgBlob = wrapMat2Blob(image);
    infer_request.SetBlob(input_name, imgBlob);
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 7. Do inference --------------------------------------------------------
    infer_request.Infer();
    // -----------------------------------------------------------------------------------------------------

	// --------------------------- 8. Process output ------------------------------------------------------
    const Blob::Ptr output_blob = infer_request.GetBlob(output_name);
    MemoryBlob::CPtr moutput = as<MemoryBlob>(output_blob);
    auto moutputHolder = moutput->rmap();
    const float *output = moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type *>();
    std::vector<float> probs;
    std::vector<cv::Rect> boxes;

    float score = 0;
    float cls = 0;
    float id = 0;
	cv::Mat result(image);

    for (int i=0; i < numPred; i++) {
        score = output[i*7+2];
		cls = output[i*7+1];
		id = output[i*7];
		std::cout<<score<<std::endl;
		if (id >= 0 && score > 0.023) {
			cv::rectangle(result, cv::Point(output[i*7+3] * w, output[i*7+4] * h), cv::Point(output[i*7+5] * w, output[i*7+6] * h), cv::Scalar(0, 255, 0));
	        //boxes.push_back(cv::Rect(output[i*7+3]*w, output[i*7+4]*h,
			//(output[i*7+5]-output[i*7+3])*w, (output[i*7+6]-output[i*7+4])*h));
			//boxes.push_back(cv::Rect(1, 1, 100, 100));
		}
	}
    //for (int i = 0; i < boxes.size(); i++) {
		//cv::rectangle(result, (10,10), (100, 100),/*boxes[i],*/ cv::Scalar(0, 255, 0));
	//}
	// -----------------------------------------------------------------------------------------------------

	// --------------------------- 9. Saving an image ------------------------------------------------------
	bool check = imwrite("../result.jpg", result);
    if (check == false) {
		std::cout << "Mission - Saving the image, FAILED" << std::endl;
	}
	// -----------------------------------------------------------------------------------------------------
	
	return 0;
}