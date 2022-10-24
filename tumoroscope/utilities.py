"""Library-wide utilities."""


class ErrorMessage:
    """Helper class for constructing informative error messages."""

    def __init__(self) -> None:
        """Create an ErrorMessage object."""
        self._msg: str | None = None

    def __call__(self, new_msg: str) -> None:
        """Contribute to an error message.

        Args:
            new_msg (str): New error add as a new line to the error message.
        """
        if self._msg is None:
            self._msg = new_msg
        else:
            self._msg += "\n" + new_msg

    @property
    def final_message(self) -> str | None:
        """If any messages have been contributed, return the full error message."""
        return self._msg
