#include "qwen.h"

#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { \
    fprintf(stderr, "第 %d 行断言失败：%s；%s\n", __LINE__, #condition, qt_last_error()); \
    return 1; \
} } while (0)

int main(void) {
    struct qt_init_params params;
    qt_init_default_params(&params);
    CHECK(QT_ABI_VERSION == 6);
    CHECK(params.n_threads == 0);
    CHECK(params.backend == NULL);

    struct qt_backend_info info = {0};
    CHECK(qt_backend_info(NULL, &info) == QT_STATUS_INVALID_PARAMS);
    CHECK(qt_backend_device_info(-1, &info) == QT_STATUS_INVALID_PARAMS);
    CHECK(qt_backend_device_info(0, NULL) == QT_STATUS_INVALID_PARAMS);
    int count = qt_backend_device_count();
    CHECK(count > 0);
    int cpu_count = 0;
    for (int index = 0; index < count; ++index) {
        CHECK(qt_backend_device_info(index, &info) == QT_STATUS_OK);
        CHECK(info.backend_name != NULL && info.backend_name[0] != '\0');
        CHECK(info.device_name != NULL && info.device_name[0] != '\0');
        CHECK(info.device_description != NULL);
        CHECK(info.n_threads == 0);
        if (info.device_type == QT_BACKEND_DEVICE_CPU) {
            CHECK(strcmp(info.device_name, "CPU") == 0);
            ++cpu_count;
        }
    }
    CHECK(cpu_count == 1);
    CHECK(qt_backend_device_info(count, &info) == QT_STATUS_INVALID_PARAMS);
    CHECK(qt_backend_probe("CPU", 1, &info) == QT_STATUS_OK);
    CHECK(info.n_threads == 1);
    CHECK(strcmp(info.backend_name, "CPU") == 0);
    CHECK(strcmp(info.device_name, "CPU") == 0);
    CHECK(info.device_type == QT_BACKEND_DEVICE_CPU);
    CHECK(qt_backend_probe("CPU", 8, &info) == QT_STATUS_OK);
    CHECK(info.n_threads == 8);
    CHECK(qt_backend_probe(NULL, 0, &info) == QT_STATUS_OK);
    CHECK(info.n_threads > 0);
    CHECK(qt_backend_probe("CPU", -1, &info) == QT_STATUS_INVALID_PARAMS);
    CHECK(qt_backend_probe("CPU", 1, NULL) == QT_STATUS_INVALID_PARAMS);
    CHECK(qt_backend_probe("", 1, &info) == QT_STATUS_INVALID_PARAMS);
    CHECK(qt_backend_probe("不存在的设备", 1, &info) == QT_STATUS_INVALID_PARAMS);
    CHECK(strstr(qt_last_error(), "不存在的设备") != NULL);

    /* 无效配置必须在打开模型前拒绝，不能靠不存在的文件碰巧失败。 */
    params.talker_path = "不存在的模型.gguf";
    params.codec_path = "不存在的编解码器.gguf";
    params.n_threads = -1;
    CHECK(qt_init(&params) == NULL);
    CHECK(strstr(qt_last_error(), "n_threads") != NULL);
    params.n_threads = 1;
    params.backend = "不存在的设备";
    CHECK(qt_init(&params) == NULL);
    CHECK(strstr(qt_last_error(), params.backend) != NULL);
    params.backend = "";
    CHECK(qt_init(&params) == NULL);
    CHECK(strstr(qt_last_error(), "backend") != NULL);
    return 0;
}
