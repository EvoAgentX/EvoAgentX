
# Messages structure:
## Setup phase
### Registed when {WS_BASE_URL}/project/{project_short_id}/regist
### Message passing in
```json
{
  "type": "setup",
  "data":{
    "project_short_id": <project_short_id>
  }
}
```

### Message passing out

#### Regist API respond
```json
{
  "success": "<true or false>",
  "error": <not empty if not successful>
  
}
```

#### Connection Confirmation
```json
{
  "type": "setup-log",
  "data":{
    "workflow_id": null,
    "content": "WebSocket connection established",
    "result": null
  }
}
```

#### Setup Start
```json
{
  "type": "setup-log",
  "data":{
    "workflow_id": null,
    "content": "Setup start...",
    "result": null
  }
}
```

#### Workflow Database Update Message
```json
{
  "type": "setup-log",
  "data":{
    "workflow_id": "<workflow_id>",
    "content": "<workflow_id> updates database status to: <uninitialized if just extracted, pending if generation complete>",
    "result": null
  }
}
```

#### Log Generation Messages
```json
{
  "type": "setup-log",
  "data":{
    "workflow_id": "<workflow_id>",
    "content": "<raw_generation_messages>",
    "result": null
  }
}
```

#### Completion
```json
{
  "type": "setup-complete",
  "data":{
    "workflow_id": "<workflow_id>",
    "content": "Workflow <workflow_id> generated...",
    "result": <workflow_graph_dictionary, not the workflow info>
  }
}
```

#### Error Messages
{
  "type": "setup-complete",
  "data":{
    "workflow_id": "<workflow_id>" or null,
    "content": "<error_description>: <raw_error_message>",
    "result": null
  }
}


## execution Phase
### Message passing in
Not through the socket, through the POST {BASE_URL}/workflow/{workflow_id}/execute
with {"inputs": <input_dictionary>}

### Message passing out while executing
```json
{
  "type": "runtime-log",
  "data":{
    "workflow_id": "<workflow_id>",
    "content": "<raw_execution_messages>",
    "result": null
  }
}
```

# Phases:
## Project Initialization
### Initial Project Registing
- Triggered from FastAPI port {WS_BASE_URL}/project/{project_short_id}/regist
- Passing in a socket and we store it with the project id
- Return error messages 

### Project Setup
- Triggered when the socket receives a signal
- Keeps feeding back messages from the given socket
- Sending the result through the socket 

### Key notes:
- The socket will not be closed after the setup, they will be stored with their short project id

## Workflow Execution
- Triggered from FastAPI port POST {BASE_URL}/workflow/{workflow_id}/execute
- Retrieve the socket
- Sending execution messages through the socket
- Sending the execution result through the socket
- Sending the execution result through the API
- If cannot retrieve the socket, just keeps running without sending running messages

# Log extraction
## Key points
- The current system use an global logger to display everything in the cmd without any distinguish identifier to indicate which workflow it belong to
- We should create separate local loggers that only logs the workflow it belongs to
- We should manage to let the workflow use it in the generation and in the execution
- We create separate loggers for each individual workflow generation or workflow execution
- We will create running logs using raw returns from those local loggers and return them using the socket we mentioned above.

