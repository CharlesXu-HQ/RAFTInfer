#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum RaftInferStatusCode {
  RAFTINFER_STATUS_OK = 0,
  RAFTINFER_STATUS_INVALID_ARGUMENT = 1,
  RAFTINFER_STATUS_UNAVAILABLE = 2,
  RAFTINFER_STATUS_CUDA_ERROR = 3,
  RAFTINFER_STATUS_INTERNAL = 4,
  RAFTINFER_STATUS_UNSUPPORTED = 5,
  RAFTINFER_STATUS_RESOURCE_EXHAUSTED = 6
} RaftInferStatusCode;

typedef struct RaftInferStatus {
  RaftInferStatusCode code;
  const char* message;
} RaftInferStatus;

#ifdef __cplusplus
}
#endif
