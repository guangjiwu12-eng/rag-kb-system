"""业务异常：统一携带 HTTP 状态码与稳定的错误码，便于前端和调用方处理。"""

from __future__ import annotations


class AppError(Exception):
    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsupportedFileType(AppError):
    status_code = 415
    code = "unsupported_file_type"


class EmptyFile(AppError):
    status_code = 400
    code = "empty_file"


class FileTooLarge(AppError):
    status_code = 413
    code = "file_too_large"


class InvalidFileContent(AppError):
    status_code = 400
    code = "invalid_file_content"


class NoTextExtracted(AppError):
    status_code = 422
    code = "no_text_extracted"


class DocumentNotFound(AppError):
    status_code = 404
    code = "document_not_found"


class InvalidQuestion(AppError):
    status_code = 400
    code = "invalid_question"


class InvalidRetrievalMode(AppError):
    status_code = 400
    code = "invalid_retrieval_mode"


class Unauthorized(AppError):
    status_code = 401
    code = "unauthorized"


class ModelNotConfigured(AppError):
    status_code = 503
    code = "model_not_configured"


class ModelInvocationError(AppError):
    status_code = 502
    code = "model_invocation_failed"
