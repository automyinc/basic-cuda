/*
 * CUDA_Bridge.h
 *
 *  Created on: Jan 13, 2018
 *      Author: mad
 */

#ifndef INCLUDE_BASIC_CUDA_BRIDGE_H_
#define INCLUDE_BASIC_CUDA_BRIDGE_H_

#include <automy/basic_cuda/CUDA_BridgeBase.hxx>
#include <automy/basic/GenericBuffer.h>


namespace automy {
namespace basic_cuda {

class CUDA_Bridge : public CUDA_BridgeBase {
public:
	CUDA_Bridge(const std::string& _vnx_name);
	
protected:
	void main() override;
	
	void handle(std::shared_ptr<const basic::ImageFrame> value, std::shared_ptr<const vnx::Sample> sample) override;
	
//	void handle(std::shared_ptr<const PointCloud> value, std::shared_ptr<const vnx::Sample> sample) override;
	
//	void handle(std::shared_ptr<const CUDA_PointCloud> value, std::shared_ptr<const vnx::Sample> sample) override;
	
	std::shared_ptr<basic::ImageFrame> upload_frame(std::shared_ptr<const basic::ImageFrame> value);
	
	std::shared_ptr<basic::ImageFrame> download_frame(std::shared_ptr<const basic::ImageFrame> value);
	
	void update();
	
	void print_stats();
	
private:
	int64_t num_download_bytes = 0;
	int64_t num_upload_bytes = 0;
	
	std::set<std::string> auto_topic_set;
	
	std::map<size_t, basic::GenericBuffer> buffers;
	
};


} // basic_cuda
} // automy

#endif /* INCLUDE_BASIC_CUDA_BRIDGE_H_ */
