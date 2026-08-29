class AudioError(Exception):
    """A refusal with an HTTP meaning, so the UI and the API report the same outcome.

    400 for a request rejected before anything loads, 499 for an interrupt, 500 for a real failure.
    """

    def __init__(self, msg: str, code: int = 400):
        super().__init__(msg)
        self.msg = msg
        self.code = code
