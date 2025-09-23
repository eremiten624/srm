#include "srm/autoaim/detector-armor.h"
#include "tensorrt/NvInferRuntime.h"
#include <cuda/cuda_runtime.h>  // 显式包含CUDA运行时头文件
#include <cuda/cublas_v2.h>     // 包含CUDA BLAS库
#include <fstream>
#include <memory>
#include <vector>
#include <algorithm>
#include <numeric>

namespace srm::autoaim {

bool ArmorDetector::Initialize() {
  /// 初始化yolo
#if defined(__APPLE__)
  std::string net_type = "coreml";
#elif defined(__linux__)
  std::string net_type = "tensorrt";
#endif
  yolo_.reset(nn::CreateYolo(net_type));
  std::string prefix = "nn.yolo.armor";
  const auto model_path = cfg.Get<std::string>({prefix, net_type});
  const auto class_num = cfg.Get<int>({prefix, "class_num"});
  const auto point_num = cfg.Get<int>({prefix, "point_num"});
  if (!yolo_->Initialize(model_path, class_num, point_num)) {
    LOG(ERROR) << "Failed to load armor nerual network.";
    return false;
  }
  return true;
}

bool ArmorDetector::Run(cv::Mat REF_IN image, ArmorPtrList REF_OUT armor_list) const {
  ///请补全

  class YoloTensorRT : public nn::Yolo {
public:
    YoloTensorRT() : engine_(nullptr), context_(nullptr), runtime_(nullptr) {}
    ~YoloTensorRT() override {
        if (context_) context_->destroy();
        if (engine_) engine_->destroy();
        if (runtime_) runtime_->destroy();
        for (auto& buf : buffers_) {
            cudaFree(buf);
        }
    }

    bool Initialize(std::string REF_IN model_file, int num_classes, int num_points) override {
        num_classes_ = num_classes;
        num_points_ = num_points;

        // 读取模型文件
        std::ifstream file(model_file, std::ios::binary);
        if (!file.good()) {
            return false;
        }

        // 读取引擎数据
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);

        // 创建运行时和引擎
        runtime_ = nvinfer1::createInferRuntime(gLogger);
        if (!runtime_) return false;

        engine_ = runtime_->deserializeCudaEngine(engine_data.data(), size);
        if (!engine_) return false;

        // 创建上下文
        context_ = engine_->createExecutionContext();
        if (!context_) return false;

        // 获取输入输出信息
        input_index_ = engine_->getBindingIndex("images");
        output_index_ = engine_->getBindingIndex("output0");

        // 获取输入尺寸
        nvinfer1::Dims input_dims = engine_->getBindingDimensions(input_index_);
        input_h_ = input_dims.d[2];
        input_w_ = input_dims.d[3];

        // 分配输入输出缓冲区
        buffers_.resize(2);
        size_t input_size = 3 * input_h_ * input_w_ * sizeof(float);
        cudaMalloc(&buffers_[input_index_], input_size);

        nvinfer1::Dims output_dims = engine_->getBindingDimensions(output_index_);
        size_t output_size = 1;
        for (int i = 0; i < output_dims.nbDims; ++i) {
            output_size *= output_dims.d[i];
        }
        output_size *= sizeof(float);
        cudaMalloc(&buffers_[output_index_], output_size);

        // 分配CPU输出缓冲区
        output_data_size_ = output_size / sizeof(float);
        output_data_ = new float[output_data_size_];

        return true;
    }

    std::vector<nn::Objects> Run(cv::Mat image) override {
        if (!engine_ || !context_) return {};

        // 预处理图像
        cv::Mat input_image;
        float ro, dw, dh;
        LetterBox(image, ro, dw, dh);

        // 转换为RGB并归一化
        cv::cvtColor(input_image, input_image, cv::COLOR_BGR2RGB);
        input_image.convertTo(input_image, CV_32FC3, 1.0 / 255.0);

        // 转换为CHW格式
        std::vector<float> input_data(3 * input_h_ * input_w_);
        float* data_ptr = input_data.data();
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < input_h_; ++h) {
                for (int w = 0; w < input_w_; ++w) {
                    *data_ptr++ = input_image.at<cv::Vec3f>(h, w)[c];
                }
            }
        }

        // 复制数据到GPU
        cudaMemcpy(buffers_[input_index_], input_data.data(),
                  input_data.size() * sizeof(float), cudaMemcpyHostToDevice);

        // 执行推理
        context_->executeV2(buffers_.data());

        // 复制结果到CPU
        cudaMemcpy(output_data_, buffers_[output_index_],
                  output_data_size_ * sizeof(float), cudaMemcpyDeviceToHost);

        // 处理输出
        std::vector<nn::Objects> objs;
        GetObjects(objs);

        // 调整坐标到原始图像尺寸
        for (auto& obj : objs) {
            obj.x1 = (obj.x1 - dw) / ro;
            obj.y1 = (obj.y1 - dh) / ro;
            obj.x2 = (obj.x2 - dw) / ro;
            obj.y2 = (obj.y2 - dh) / ro;

            // 调整关键点坐标
            for (auto& pt : obj.pts) {
                pt.x = (pt.x - dw) / ro;
                pt.y = (pt.y - dh) / ro;
            }
        }

        return objs;
    }

protected:
    void LetterBox(cv::Mat REF_OUT image, float REF_OUT ro, float REF_OUT dw, float REF_OUT dh) override {
        int h = image.rows;
        int w = image.cols;

        // 计算缩放比例
        ro = std::min((float)input_h_ / h, (float)input_w_ / w);
        int nh = h * ro;
        int nw = w * ro;

        // 计算填充
        dw = (input_w_ - nw) / 2.0f;
        dh = (input_h_ - nh) / 2.0f;

        // 缩放图像
        cv::resize(image, image, cv::Size(nw, nh));

        // 填充边框
        cv::copyMakeBorder(image, image, dh, input_h_ - nh - dh,
                          dw, input_w_ - nw - dw, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    }

    void GetObjects(std::vector<nn::Objects> REF_OUT objs) override {
        objs.clear();

        // YOLO输出格式: [x, y, w, h, conf, cls, (points...)]
        int elements_per_object = 5 + num_classes_ + 2 * num_points_;
        int num_objects = output_data_size_ / elements_per_object;

        for (int i = 0; i < num_objects; ++i) {
            float* data = output_data_ + i * elements_per_object;

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];
            float conf = data[4];

            // 跳过低置信度目标
            if (conf < box_conf_thresh_) continue;

            // 找到置信度最高的类别
            float max_cls_conf = 0;
            long cls = 0;
            for (int c = 0; c < num_classes_; ++c) {
                if (data[5 + c] > max_cls_conf) {
                    max_cls_conf = data[5 + c];
                    cls = c;
                }
            }

            // 计算最终置信度
            float final_conf = conf * max_cls_conf;
            if (final_conf < box_conf_thresh_) continue;

            // 转换中心点坐标为左上角和右下角坐标
            nn::Objects obj;
            obj.x1 = x - w / 2;
            obj.y1 = y - h / 2;
            obj.x2 = x + w / 2;
            obj.y2 = y + h / 2;
            obj.prob = final_conf;
            obj.cls = cls;

            // 提取关键点
            for (int p = 0; p < num_points_; ++p) {
                float px = data[5 + num_classes_ + 2 * p];
                float py = data[5 + num_classes_ + 2 * p + 1];
                obj.pts.emplace_back(px, py);
            }

            objs.push_back(obj);
        }

        // 执行非极大值抑制
        NMS(objs);
    }

    void NMS(std::vector<nn::Objects> REF_OUT objs) override {
        if (objs.empty()) return;

        // 按类别分组
        std::unordered_map<long, std::vector<nn::Objects>> class_objs;
        for (const auto& obj : objs) {
            class_objs[obj.cls].push_back(obj);
        }

        objs.clear();

        // 对每个类别执行NMS
        for (auto& [cls, cls_objects] : class_objs) {
            // 按置信度排序
            std::sort(cls_objects.begin(), cls_objects.end(),
                      [](const nn::Objects& a, const nn::Objects& b) {
                          return a.prob > b.prob;
                      });

            // 限制最大处理数量
            if (cls_objects.size() > max_nms_) {
                cls_objects.resize(max_nms_);
            }

            // 执行NMS
            std::vector<bool> keep(cls_objects.size(), true);
            for (int i = 0; i < cls_objects.size(); ++i) {
                if (!keep[i]) continue;

                for (int j = i + 1; j < cls_objects.size(); ++j) {
                    if (!keep[j]) continue;

                    // 计算IOU
                    float ix1 = std::max(cls_objects[i].x1, cls_objects[j].x1);
                    float iy1 = std::max(cls_objects[i].y1, cls_objects[j].y1);
                    float ix2 = std::min(cls_objects[i].x2, cls_objects[j].x2);
                    float iy2 = std::min(cls_objects[i].y2, cls_objects[j].y2);

                    float area_i = (cls_objects[i].x2 - cls_objects[i].x1) *
                                  (cls_objects[i].y2 - cls_objects[i].y1);
                    float area_j = (cls_objects[j].x2 - cls_objects[j].x1) *
                                  (cls_objects[j].y2 - cls_objects[j].y1);
                    float area_inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
                    float iou = area_inter / (area_i + area_j - area_inter);

                    if (iou > iou_thresh_) {
                        keep[j] = false;
                    }
                }
            }

            // 保留结果
            for (int i = 0; i < cls_objects.size(); ++i) {
                if (keep[i]) {
                    objs.push_back(cls_objects[i]);
                }
            }
        }
    }

private:
    // TensorRT相关成员
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    nvinfer1::IRuntime* runtime_;
    std::vector<void*> buffers_;
    int input_index_;
    int output_index_;
    size_t output_data_size_;

    // TensorRT日志类
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            // 只打印错误信息
            if (severity <= Severity::kERROR) {
                std::cout << msg << std::endl;
            }
        }
    } gLogger;
};
  std::vector<nn::Objects> Objects_list = yolo_->Run(image);
  return true;
}

}  // namespace srm::autoaim