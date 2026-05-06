import io
import logging
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, List

import websockets

logger = logging.getLogger(__name__)


class MsgType(IntEnum):
    """Message type enumeration"""

    Invalid = 0
    FullClientRequest = 0b1
    AudioOnlyClient = 0b10
    FullServerResponse = 0b1001
    AudioOnlyServer = 0b1011
    FrontEndResultServer = 0b1100
    Error = 0b1111

    # Alias
    ServerACK = AudioOnlyServer

    def __str__(self) -> str:
        return self.name if self.name else f"MsgType({self.value})"


class MsgTypeFlagBits(IntEnum):
    """Message type flag bits"""

    NoSeq = 0  # Non-terminal packet with no sequence
    PositiveSeq = 0b1  # Non-terminal packet with sequence > 0
    LastNoSeq = 0b10  # Last packet with no sequence
    NegativeSeq = 0b11  # Last packet with sequence < 0
    WithEvent = 0b100  # Payload contains event number (int32)


class VersionBits(IntEnum):
    """Version bits"""

    Version1 = 1
    Version2 = 2
    Version3 = 3
    Version4 = 4


class HeaderSizeBits(IntEnum):
    """Header size bits"""

    HeaderSize4 = 1
    HeaderSize8 = 2
    HeaderSize12 = 3
    HeaderSize16 = 4


class SerializationBits(IntEnum):
    """Serialization method bits"""

    Raw = 0
    JSON = 0b1
    Thrift = 0b11
    Custom = 0b1111


class CompressionBits(IntEnum):
    """Compression method bits"""

    None_ = 0
    Gzip = 0b1
    Custom = 0b1111


class EventType(IntEnum):
    """Event type enumeration"""

    None_ = 0  # Default event

    # 1 ~ 49 Upstream Connection events
    StartConnection = 1
    StartTask = 1  # Alias of StartConnection
    FinishConnection = 2
    FinishTask = 2  # Alias of FinishConnection

    # 50 ~ 99 Downstream Connection events
    ConnectionStarted = 50  # Connection established successfully
    TaskStarted = 50  # Alias of ConnectionStarted
    ConnectionFailed = 51  # Connection failed (possibly due to authentication failure)
    TaskFailed = 51  # Alias of ConnectionFailed
    ConnectionFinished = 52  # Connection ended
    TaskFinished = 52  # Alias of ConnectionFinished

    # 100 ~ 149 Upstream Session events
    StartSession = 100
    CancelSession = 101
    FinishSession = 102

    # 150 ~ 199 Downstream Session events
    SessionStarted = 150
    SessionCanceled = 151
    SessionFinished = 152
    SessionFailed = 153
    UsageResponse = 154  # Usage response
    ChargeData = 154  # Alias of UsageResponse

    # 200 ~ 249 Upstream general events
    TaskRequest = 200
    UpdateConfig = 201

    # 250 ~ 299 Downstream general events
    AudioMuted = 250

    # 300 ~ 349 Upstream TTS events
    SayHello = 300

    # 350 ~ 399 Downstream TTS events
    TTSSentenceStart = 350
    TTSSentenceEnd = 351
    TTSResponse = 352
    TTSEnded = 359
    PodcastRoundStart = 360
    PodcastRoundResponse = 361
    PodcastRoundEnd = 362

    # 450 ~ 499 Downstream ASR events
    ASRInfo = 450
    ASRResponse = 451
    ASREnded = 459

    # 500 ~ 549 Upstream dialogue events
    ChatTTSText = 500  # (Ground-Truth-Alignment) text for speech synthesis

    # 550 ~ 599 Downstream dialogue events
    ChatResponse = 550
    ChatEnded = 559

    # 650 ~ 699 Downstream dialogue events
    # Events for source (original) language subtitle
    SourceSubtitleStart = 650
    SourceSubtitleResponse = 651
    SourceSubtitleEnd = 652
    # Events for target (translation) language subtitle
    TranslationSubtitleStart = 653
    TranslationSubtitleResponse = 654
    TranslationSubtitleEnd = 655

    def __str__(self) -> str:
        return self.name if self.name else f"EventType({self.value})"


@dataclass
class Message:
    """Message object

    Message format:
    0                 1                 2                 3
    | 0 1 2 3 4 5 6 7 | 0 1 2 3 4 5 6 7 | 0 1 2 3 4 5 6 7 | 0 1 2 3 4 5 6 7 |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |    Version      |   Header Size   |     Msg Type    |      Flags      |
    |   (4 bits)      |    (4 bits)     |     (4 bits)    |     (4 bits)    |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    | Serialization   |   Compression   |           Reserved                |
    |   (4 bits)      |    (4 bits)     |           (8 bits)                |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                                                                       |
    |                   Optional Header Extensions                          |
    |                     (if Header Size > 1)                              |
    |                                                                       |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    |                                                                       |
    |                           Payload                                     |
    |                      (variable length)                                |
    |                                                                       |
    +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    """

    version: VersionBits = VersionBits.Version1
    header_size: HeaderSizeBits = HeaderSizeBits.HeaderSize4
    type: MsgType = MsgType.Invalid
    flag: MsgTypeFlagBits = MsgTypeFlagBits.NoSeq
    serialization: SerializationBits = SerializationBits.JSON
    compression: CompressionBits = CompressionBits.None_

    event: EventType = EventType.None_
    session_id: str = ""
    connect_id: str = ""
    sequence: int = 0
    error_code: int = 0

    payload: bytes = b""

    @classmethod
    def from_bytes(cls, data: bytes) -> "Message":
        """Create message object from bytes"""
        if len(data) < 3:
            raise ValueError(
                f"Data too short: expected at least 3 bytes, got {len(data)}"
            )

        type_and_flag = data[1]
        msg_type = MsgType(type_and_flag >> 4)
        flag = MsgTypeFlagBits(type_and_flag & 0b00001111)

        msg = cls(type=msg_type, flag=flag)
        msg.unmarshal(data)
        return msg

    def marshal(self) -> bytes:
        """Serialize message to bytes"""
        buffer = io.BytesIO()

        # Write header
        header = [
            (self.version << 4) | self.header_size,
            (self.type << 4) | self.flag,
            (self.serialization << 4) | self.compression,
        ]

        header_size = 4 * self.header_size
        if padding := header_size - len(header):
            header.extend([0] * padding)

        buffer.write(bytes(header))

        # Write other fields
        writers = self._get_writers()
        for writer in writers:
            writer(buffer)

        return buffer.getvalue()

    def unmarshal(self, data: bytes) -> None:
        """Deserialize message from bytes"""
        buffer = io.BytesIO(data)

        # Read version and header size
        version_and_header_size = buffer.read(1)[0]
        self.version = VersionBits(version_and_header_size >> 4)
        self.header_size = HeaderSizeBits(version_and_header_size & 0b00001111)

        # Skip second byte
        buffer.read(1)

        # Read serialization and compression methods
        serialization_compression = buffer.read(1)[0]
        self.serialization = SerializationBits(serialization_compression >> 4)
        self.compression = CompressionBits(serialization_compression & 0b00001111)

        # Skip header padding
        header_size = 4 * self.header_size
        read_size = 3
        if padding_size := header_size - read_size:
            buffer.read(padding_size)

        # Read other fields
        readers = self._get_readers()
        for reader in readers:
            reader(buffer)

        # Check for remaining data
        remaining = buffer.read()
        if remaining:
            raise ValueError(f"Unexpected data after message: {remaining}")

    def _get_writers(self) -> List[Callable[[io.BytesIO], None]]:
        """Get list of writer functions"""
        writers = []

        if self.flag == MsgTypeFlagBits.WithEvent:
            writers.extend([self._write_event, self._write_session_id])

        if self.type in [
            MsgType.FullClientRequest,
            MsgType.FullServerResponse,
            MsgType.FrontEndResultServer,
            MsgType.AudioOnlyClient,
            MsgType.AudioOnlyServer,
        ]:
            if self.flag in [MsgTypeFlagBits.PositiveSeq, MsgTypeFlagBits.NegativeSeq]:
                writers.append(self._write_sequence)
        elif self.type == MsgType.Error:
            writers.append(self._write_error_code)
        else:
            raise ValueError(f"Unsupported message type: {self.type}")

        writers.append(self._write_payload)
        return writers

    def _get_readers(self) -> List[Callable[[io.BytesIO], None]]:
        """Get list of reader functions"""
        readers = []

        if self.type in [
            MsgType.FullClientRequest,
            MsgType.FullServerResponse,
            MsgType.FrontEndResultServer,
            MsgType.AudioOnlyClient,
            MsgType.AudioOnlyServer,
        ]:
            if self.flag in [MsgTypeFlagBits.PositiveSeq, MsgTypeFlagBits.NegativeSeq]:
                readers.append(self._read_sequence)
        elif self.type == MsgType.Error:
            readers.append(self._read_error_code)
        else:
            raise ValueError(f"Unsupported message type: {self.type}")

        if self.flag == MsgTypeFlagBits.WithEvent:
            readers.extend(
                [self._read_event, self._read_session_id, self._read_connect_id]
            )

        readers.append(self._read_payload)
        return readers

    def _write_event(self, buffer: io.BytesIO) -> None:
        """Write event"""
        buffer.write(struct.pack(">i", self.event))

    def _write_session_id(self, buffer: io.BytesIO) -> None:
        """Write session ID"""
        if self.event in [
            EventType.StartConnection,
            EventType.FinishConnection,
            EventType.ConnectionStarted,
            EventType.ConnectionFailed,
        ]:
            return

        session_id_bytes = self.session_id.encode("utf-8")
        size = len(session_id_bytes)
        if size > 0xFFFFFFFF:
            raise ValueError(f"Session ID size ({size}) exceeds max(uint32)")

        buffer.write(struct.pack(">I", size))
        if size > 0:
            buffer.write(session_id_bytes)

    def _write_sequence(self, buffer: io.BytesIO) -> None:
        """Write sequence number"""
        buffer.write(struct.pack(">i", self.sequence))

    def _write_error_code(self, buffer: io.BytesIO) -> None:
        """Write error code"""
        buffer.write(struct.pack(">I", self.error_code))

    def _write_payload(self, buffer: io.BytesIO) -> None:
        """Write payload"""
        size = len(self.payload)
        if size > 0xFFFFFFFF:
            raise ValueError(f"Payload size ({size}) exceeds max(uint32)")

        buffer.write(struct.pack(">I", size))
        buffer.write(self.payload)

    def _read_event(self, buffer: io.BytesIO) -> None:
        """Read event"""
        event_bytes = buffer.read(4)
        if event_bytes:
            self.event = EventType(struct.unpack(">i", event_bytes)[0])

    def _read_session_id(self, buffer: io.BytesIO) -> None:
        """Read session ID"""
        if self.event in [
            EventType.StartConnection,
            EventType.FinishConnection,
            EventType.ConnectionStarted,
            EventType.ConnectionFailed,
            EventType.ConnectionFinished,
        ]:
            return

        size_bytes = buffer.read(4)
        if size_bytes:
            size = struct.unpack(">I", size_bytes)[0]
            if size > 0:
                session_id_bytes = buffer.read(size)
                if len(session_id_bytes) == size:
                    self.session_id = session_id_bytes.decode("utf-8")

    def _read_connect_id(self, buffer: io.BytesIO) -> None:
        """Read connection ID"""
        if self.event in [
            EventType.ConnectionStarted,
            EventType.ConnectionFailed,
            EventType.ConnectionFinished,
        ]:
            size_bytes = buffer.read(4)
            if size_bytes:
                size = struct.unpack(">I", size_bytes)[0]
                if size > 0:
                    self.connect_id = buffer.read(size).decode("utf-8")

    def _read_sequence(self, buffer: io.BytesIO) -> None:
        """Read sequence number"""
        sequence_bytes = buffer.read(4)
        if sequence_bytes:
            self.sequence = struct.unpack(">i", sequence_bytes)[0]

    def _read_error_code(self, buffer: io.BytesIO) -> None:
        """Read error code"""
        error_code_bytes = buffer.read(4)
        if error_code_bytes:
            self.error_code = struct.unpack(">I", error_code_bytes)[0]

    def _read_payload(self, buffer: io.BytesIO) -> None:
        """Read payload"""
        size_bytes = buffer.read(4)
        if size_bytes:
            size = struct.unpack(">I", size_bytes)[0]
            if size > 0:
                self.payload = buffer.read(size)

    def __str__(self) -> str:
        """String representation"""
        if self.type in [MsgType.AudioOnlyServer, MsgType.AudioOnlyClient]:
            if self.flag in [MsgTypeFlagBits.PositiveSeq, MsgTypeFlagBits.NegativeSeq]:
                return f"MsgType: {self.type}, EventType:{self.event}, Sequence: {self.sequence}, PayloadSize: {len(self.payload)}"
            return f"MsgType: {self.type}, EventType:{self.event}, PayloadSize: {len(self.payload)}"
        elif self.type == MsgType.Error:
            return f"MsgType: {self.type}, EventType:{self.event}, ErrorCode: {self.error_code}, Payload: {self.payload.decode('utf-8', 'ignore')}"
        else:
            if self.flag in [MsgTypeFlagBits.PositiveSeq, MsgTypeFlagBits.NegativeSeq]:
                return f"MsgType: {self.type}, EventType:{self.event}, Sequence: {self.sequence}, Payload: {self.payload.decode('utf-8', 'ignore')}"
            return f"MsgType: {self.type}, EventType:{self.event}, Payload: {self.payload.decode('utf-8', 'ignore')}"


async def receive_message(websocket: websockets.WebSocketClientProtocol) -> Message:
    """Receive message from websocket"""
    try:
        data = await websocket.recv()
        if isinstance(data, str):
            raise ValueError(f"Unexpected text message: {data}")
        elif isinstance(data, bytes):
            msg = Message.from_bytes(data)
            logger.info(f"Received: {msg}")
            return msg
        else:
            raise ValueError(f"Unexpected message type: {type(data)}")
    except Exception as e:
        logger.error(f"Failed to receive message: {e}")
        raise


async def wait_for_event(
    websocket: websockets.WebSocketClientProtocol,
    msg_type: MsgType,
    event_type: EventType,
) -> Message:
    """Wait for specific event"""
    while True:
        msg = await receive_message(websocket)
        if msg.type != msg_type or msg.event != event_type:
            raise ValueError(f"Unexpected message: {msg}")
        if msg.type == msg_type and msg.event == event_type:
            return msg


async def full_client_request(
    websocket: websockets.WebSocketClientProtocol, payload: bytes
) -> None:
    """Send full client message"""
    msg = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.NoSeq)
    msg.payload = payload
    logger.info(f"Sending: {msg}")
    await websocket.send(msg.marshal())


async def audio_only_client(
    websocket: websockets.WebSocketClientProtocol, payload: bytes, flag: MsgTypeFlagBits
) -> None:
    """Send audio-only client message"""
    msg = Message(type=MsgType.AudioOnlyClient, flag=flag)
    msg.payload = payload
    logger.info(f"Sending: {msg}")
    await websocket.send(msg.marshal())


async def start_connection(websocket: websockets.WebSocketClientProtocol) -> None:
    """Start connection"""
    msg = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.WithEvent)
    msg.event = EventType.StartConnection
    msg.payload = b"{}"
    logger.info(f"Sending: {msg}")
    await websocket.send(msg.marshal())


async def finish_connection(websocket: websockets.WebSocketClientProtocol) -> None:
    """Finish connection"""
    msg = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.WithEvent)
    msg.event = EventType.FinishConnection
    msg.payload = b"{}"
    logger.info(f"Sending: {msg}")
    await websocket.send(msg.marshal())


async def start_session(
    websocket: websockets.WebSocketClientProtocol, payload: bytes, session_id: str
) -> None:
    """Start session"""
    msg = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.WithEvent)
    msg.event = EventType.StartSession
    msg.session_id = session_id
    msg.payload = payload
    logger.info(f"Sending: {msg}")
    await websocket.send(msg.marshal())


async def finish_session(
    websocket: websockets.WebSocketClientProtocol, session_id: str
) -> None:
    """Finish session"""
    msg = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.WithEvent)
    msg.event = EventType.FinishSession
    msg.session_id = session_id
    msg.payload = b"{}"
    logger.info(f"Sending: {msg}")
    await websocket.send(msg.marshal())


async def cancel_session(
    websocket: websockets.WebSocketClientProtocol, session_id: str
) -> None:
    """Cancel session"""
    msg = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.WithEvent)
    msg.event = EventType.CancelSession
    msg.session_id = session_id
    msg.payload = b"{}"
    logger.info(f"Sending: {msg}")
    await websocket.send(msg.marshal())


async def task_request(
    websocket: websockets.WebSocketClientProtocol, payload: bytes, session_id: str
) -> None:
    """Send task request"""
    msg = Message(type=MsgType.FullClientRequest, flag=MsgTypeFlagBits.WithEvent)
    msg.event = EventType.TaskRequest
    msg.session_id = session_id
    msg.payload = payload
    logger.info(f"Sending: {msg}")
    await websocket.send(msg.marshal())
