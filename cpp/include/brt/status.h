#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum BrtStatusCode {
  BRT_STATUS_OK = 0,
  BRT_STATUS_INVALID_ARGUMENT = 1,
  BRT_STATUS_UNAVAILABLE = 2,
  BRT_STATUS_CUDA_ERROR = 3,
  BRT_STATUS_INTERNAL = 4,
  BRT_STATUS_UNSUPPORTED = 5,
  BRT_STATUS_RESOURCE_EXHAUSTED = 6
} BrtStatusCode;

typedef struct BrtStatus {
  BrtStatusCode code;
  const char* message;
} BrtStatus;

#ifdef __cplusplus
}
#endif
