/*
 * CUDA_Bridge.cpp
 *
 *  Created on: Jan 13, 2018
 *      Author: mad
 */

#include <automy/basic_cuda/CUDA_Bridge.h>
#include <automy/basic_cuda/CUDA_ImageFrame8.hxx>
#include <automy/basic_cuda/CUDA_ImageFrame16.hxx>
#include <automy/basic_cuda/CUDA_ImageFrameF32.hxx>
#include <automy/basic/ImageFrame8.hxx>
#include <automy/basic/ImageFrame16.hxx>
#include <automy/basic/ImageFrameF32.hxx>
#include <automy/basic/MultiImageFrame.hxx>
#include <automy/math/math.h>
#include <vnx/vnx.h>

using namespace automy::math;


namespace automy {
namespace basic_cuda {

CUDA_Bridge::CUDA_Bridge(const std::string& _vnx_name) : CUDA_BridgeBase(_vnx_name) {}

void CUDA_Bridge::main() {
	for(const std::string& topic : upload) {
		subscribe(cpu_domain + "." + topic);
	}
	for(const std::string& topic : download) {
		subscribe(cuda_domain + "." + topic);
	}
	set_timer_millis(interval_ms, std::bind(&CUDA_Bridge::update, this));
	set_timer_millis(1000, std::bind(&CUDA_Bridge::print_stats, this));
	
	if(device < 0) {
		vnx::read_config("cuda.device", device);
	}
	cudaSetDevice(device);
	
	Super::main();
}

void CUDA_Bridge::handle(std::shared_ptr<const basic::ImageFrame> value, std::shared_ptr<const vnx::Sample> sample) {
	const std::string topic_name = sample->topic->get_name();
	
	if(topic_name.substr(0, cpu_domain.size()) == cpu_domain) {
		
		std::shared_ptr<basic::ImageFrame> out = upload_frame(value);
		if(out) {
			*out = *value;
			publish(out, vnx::get_topic(cuda_domain + topic_name.substr(cpu_domain.size())));
		}
	} else if(topic_name.substr(0, cuda_domain.size()) == cuda_domain) {
		
		std::shared_ptr<basic::ImageFrame> out = download_frame(value);
		if(out) {
			*out = *value;
			publish(out, vnx::get_topic(cpu_domain + topic_name.substr(cuda_domain.size())));
		}
	}
}

std::shared_ptr<basic::ImageFrame> CUDA_Bridge::upload_frame(std::shared_ptr<const basic::ImageFrame> value) {
	{
		std::shared_ptr<const basic::ImageFrame8> input = std::dynamic_pointer_cast<const basic::ImageFrame8>(value);
		if(input) {
			const size_t size = input->image.get_size();
			std::shared_ptr<CUDA_ImageFrame8> out = buffers[size].get<CUDA_ImageFrame8>();
			out->image = input->image;
			num_upload_bytes += size;
			return out;
		}
	}
	{
		std::shared_ptr<const basic::ImageFrame16> input = std::dynamic_pointer_cast<const basic::ImageFrame16>(value);
		if(input) {
			const size_t size = input->image.get_size() * 2;
			std::shared_ptr<CUDA_ImageFrame16> out = buffers[size].get<CUDA_ImageFrame16>();
			out->num_bits = input->num_bits;
			out->image = input->image;
			num_upload_bytes += size;
			return out;
		}
	}
	{
		std::shared_ptr<const basic::ImageFrameF16> input = std::dynamic_pointer_cast<const basic::ImageFrameF16>(value);
		if(input) {
			const size_t size = input->image.get_size() * 2;
			std::shared_ptr<CUDA_ImageFrameF16> out = buffers[size].get<CUDA_ImageFrameF16>();
			out->image = input->image;
			num_upload_bytes += size;
			return out;
		}
	}
	{
		std::shared_ptr<const basic::ImageFrameF32> input = std::dynamic_pointer_cast<const basic::ImageFrameF32>(value);
		if(input) {
			const size_t size = input->image.get_size() * 4;
			std::shared_ptr<CUDA_ImageFrameF32> out = buffers[size].get<CUDA_ImageFrameF32>();
			out->image = input->image;
			num_upload_bytes += size;
			return out;
		}
	}
	{
		std::shared_ptr<const basic::MultiImageFrame> input = std::dynamic_pointer_cast<const basic::MultiImageFrame>(value);
		if(input) {
			std::shared_ptr<basic::MultiImageFrame> out = MultiImageFrame::create();
			for(std::shared_ptr<const basic::ImageFrame> input_layer : input->frames) {
				std::shared_ptr<basic::ImageFrame> layer = upload_frame(input_layer);
				if(layer) {
					layer->time = input_layer->time;
					layer->format = input_layer->format;
					out->frames.push_back(layer);
				}
			}
			return out;
		}
	}
	return 0;
}

std::shared_ptr<ImageFrame> CUDA_Bridge::download_frame(std::shared_ptr<const ImageFrame> value) {
	{
		std::shared_ptr<const CUDA_ImageFrame8> input = std::dynamic_pointer_cast<const CUDA_ImageFrame8>(value);
		if(input) {
			const size_t size = input->image.get_size();
			std::shared_ptr<basic::ImageFrame8> out = buffers[size].get<basic::ImageFrame8>();
			input->image.download(out->image);
			num_download_bytes += size;
			return out;
		}
	}
	{
		std::shared_ptr<const CUDA_ImageFrame16> input = std::dynamic_pointer_cast<const CUDA_ImageFrame16>(value);
		if(input) {
			const size_t size = input->image.get_size() * 2;
			std::shared_ptr<basic::ImageFrame16> out = buffers[size].get<basic::ImageFrame16>();
			out->num_bits = input->num_bits;
			input->image.download(out->image);
			num_download_bytes += size;
			return out;
		}
	}
	{
		std::shared_ptr<const CUDA_ImageFrameF16> input = std::dynamic_pointer_cast<const CUDA_ImageFrameF16>(value);
		if(input) {
			const size_t size = input->image.get_size() * 2;
			std::shared_ptr<basic::ImageFrameF16> out = buffers[size].get<basic::ImageFrameF16>();
			input->image.download(out->image);
			num_download_bytes += size;
			return out;
		}
	}
	{
		std::shared_ptr<const CUDA_ImageFrameF32> input = std::dynamic_pointer_cast<const CUDA_ImageFrameF32>(value);
		if(input) {
			const size_t size = input->image.get_size() * 4;
			std::shared_ptr<basic::ImageFrameF32> out = buffers[size].get<basic::ImageFrameF32>();
			input->image.download(out->image);
			num_download_bytes += size;
			return out;
		}
	}
	{
		std::shared_ptr<const basic::MultiImageFrame> input = std::dynamic_pointer_cast<const basic::MultiImageFrame>(value);
		if(input) {
			std::shared_ptr<basic::MultiImageFrame> out = MultiImageFrame::create();
			for(std::shared_ptr<const basic::ImageFrame> input_layer : input->frames) {
				std::shared_ptr<basic::ImageFrame> layer = download_frame(input_layer);
				if(layer) {
					layer->time = input_layer->time;
					layer->format = input_layer->format;
					out->frames.push_back(layer);
				}
			}
			return out;
		}
	}
	return 0;
}

//void CUDA_Bridge::handle(std::shared_ptr<const PointCloud> value, std::shared_ptr<const vnx::Sample> sample) {
//	const std::string topic_name = sample->topic->get_name();
//	if(topic_name.substr(0, cpu_domain.size()) == cpu_domain) {
//		// TODO
//	}
//}

//void CUDA_Bridge::handle(std::shared_ptr<const CUDA_PointCloud> value, std::shared_ptr<const vnx::Sample> sample) {
//	const std::string topic_name = sample->topic->get_name();
//	if(topic_name.substr(0, cuda_domain.size()) == cuda_domain) {
//		std::shared_ptr<PointCloud> out = PointCloud::create();
//		out->time = value->time;
//		out->frame = value->frame;
//		out->camera_info = value->camera_info;
//		{
//			basic::ImageF32 points(value->num_points, 1, 4);
//			basic::Image8 color(value->num_points, 1, 4);
//			cuda_check(cudaMemcpy(points.get_data(), value->points.get_data(), value->num_points * 16, cudaMemcpyDeviceToHost), "cudaMemcpy() :");
//			cuda_check(cudaMemcpy(color.get_data(), value->color.get_data(), value->num_points * 4, cudaMemcpyDeviceToHost), "cudaMemcpy() :");
//
//			out->points.reserve(value->num_points);
//			for(int i = 0; i < value->num_points; ++i) {
//				const float* p_point = &points(i, 0, 0);
//				const uint8_t* p_color = &color(i, 0, 0);
//				point_t point;
//				point.error = p_point[3];
//				point.distance = p_point[0];
//				point.position = Vector3f(p_point[0], p_point[1], p_point[2]);
//				point.color = Vector4uc(p_color[0], p_color[1], p_color[2], 255);
//				out->points.push_back(point);
//			}
//			num_download_bytes += value->num_points * 20;
//		}
//		publish(out, vnx::get_topic(cpu_domain + topic_name.substr(cuda_domain.size())));
//	}
//}

void CUDA_Bridge::update() {
	std::set<std::string> upload_topic_set;
	for(const std::string& topic : upload) {
		upload_topic_set.insert(cpu_domain + "." + topic);
	}
	
	std::set<std::string> cuda_topic_set;
	auto cuda_topics = vnx::get_all_topics(vnx::get_topic(cuda_domain));
	for(std::shared_ptr<vnx::Topic> topic : cuda_topics) {
		cuda_topic_set.insert(topic->get_name());
	}
	
	auto cpu_topics = vnx::get_all_topics(vnx::get_topic(cpu_domain));
	for(std::shared_ptr<vnx::Topic> topic : cpu_topics) {
		const vnx::TopicInfo info = topic->get_info();
		const std::string cuda_topic = vnx::string_subs(info.name, cpu_domain, cuda_domain);
		if(cuda_topic_set.count(cuda_topic) == 0) {
			continue;
		}
		if(auto_topic_set.count(cuda_topic) == 0) {
			if(info.num_subscribers > 0 && upload_topic_set.count(info.name) == 0) {
				subscribe(cuda_topic);
				auto_topic_set.insert(cuda_topic);
				log(INFO).out << "Enable download of '" << cuda_topic << "'";
			}
		} else {
			if(info.num_subscribers == 0) {
				unsubscribe(cuda_topic);
				auto_topic_set.erase(cuda_topic);
				log(INFO).out << "Disable download of '" << cuda_topic << "'";
			}
		}
	}
}

void CUDA_Bridge::print_stats() {
	log(INFO).out << (num_upload_bytes/1024/1024) << " MB/s upload, "
				<< (num_download_bytes/1024/1024) << " MB/s download";
	num_download_bytes = 0;
	num_upload_bytes = 0;
}


} // basic_cuda
} // automy
