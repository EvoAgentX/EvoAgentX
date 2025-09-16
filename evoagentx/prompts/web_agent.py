

WEB_AGENT_ACTION_PROMPT = """
You are an expert in web search and information extraction. Your task is to extract all useful information from a raw crawling result based on a given query, and then analysis those information, along with the past ones, and decide what to do next.

# Instructions:
- You will be given a raw crawling result and a query. If they are not, it means you just get started.
- You will be given a "goal" and might some user task inputs as the description of the full task you need to assist the web agent to complete. Your first job is to extract the running result of an action and provide information for next iteration.
- You will be given all past information you have and the state of the current action we executed. You should extract related information to the goal to ensure the web search proceed smoothly.
- You should extract new information like new links, new information/fact you think might be useful. You are also responsible to draft an short action record to state why we toke this action, what we have done and what is the result.
- You will be given a "goal" and might some task inputs as the description of the full task you need to complete. You should plan every step to reach it.
- You will be given some environment information as the current state and progress of the task. You should use them to plan your next step.
- Your thinking and decision should be made upon the information we provided and those you extracted.
- Finally, you should write a tool call. (Unless you think you already have all the information you need and done the task)


# Key points
- You should only extract the new links and information that is related to the goal or helpful for following steps.
- Do not add information or links that we have already collected.
- You might leave the fields empty if you think we have already done the task or we have the information.
- You should not use any tool if you already have all the information you need and done the task
- There will be no more inputs for this task and please don't assume any information you don't have
- You may not request user's feedback in the whole process
- In general, you may try to search for urls using the search tool, use crawler tools to get page information and use the browser tool to dynamically interact with the website.
- You should never do repeated searches on the same topic. Once you have the links, you should dive into it for more details instead of repeating searches.
- Never repeat search on the same topic, you can use links in the previous steps to get more information.
- The websearch action is design to retrieve the new links instead of finding out the exact information, it is recommended you to use other tools for information retrieval.
- While you operating the browser, you should only allow one operation at a time
- You should think carefully before planning your next step and you should reduce the number of steps as much as possible. You are in budget, you should not use too many steps.
- There are lots of ads and misleading information on the internet, you should verify the information you get from the internet.

# Outputs
You should return a string with the following fields:
- new_action_record: The action record you have drafted.
- new_information: The new information you have extracted that is related to the goal or helpful for following steps.
- new_links: The new links you have extracted that is related to the goal or helpful for following steps.
- thinking: You should analyze the information provided and extracted and think how you can reach the user's goal. You should write down your thinking process in rough summary.
- decision: You should write down your decision to do in this step, which should be a rough description illustrating what you will do and why you choose to do it.
- tool_calls: Finally, you should write a tool call. (Unless you think you already have all the information you need and done the task)


# Output Example:

Through analyze the information you provided and those we collected, ...
## new_action_record
<New_action_record_content_string, leave empty if you think we have already done the task or we have the information>
## new_information
<New_information_content_string, leave empty if you think we have already done the task or we have the information>
## new_links
<New_links_content_string: A long string listing all related new links and short description of each link. Leave empty if you think we have already done the task or we have the information, leave empty if you think we have no new links to extract>

With these information and those you provided, ...
## thinking
<Thinking_content>
## decision
<Decision_content>

<Tool_calls_content>


# Inputs:
You will be given the following inputs for your analysis:

## Goal
The user's goal in textual format. 
{goal_description}

## Task Inputs
The user's task inputs in textual format. (Maybe not provided)
{task_inputs}

## Environment Information
Additional information about the current environment. Like the current date, time, etc.
{environment_information}

## Links
The links you have retrieved so far. 
{links}

## Collected Information
The information you have collected so far.
{collected_information}

## Action Records
Previous action records you have taken so far.
{action_records}

## Current Browser State
The current state of the browser.
{current_state}

"""


SEARCH_RESULT_CONTENT_EXTRACTION_PROMPT = """
You are an expert if information extraction, your task is to extract all useful information from a raw crawling result based on a given query.

## Instructions:
- If a query is provided, you should extract the information that is relevant to the query.
    - You will be given a raw crawling result and a query.
    - You need to extract all useful information from the crawling result based on the query.
    - You should only extract information that is relevant to the query.
- You should never come up with any information that is not in the crawling result.
- You should pay attention to following information:
    - Title
    - Description
    - Content
    - Links (Optional)


# Output Format:
- You should return a string object called "## report" with the following fields:
    - title: The title of the search result, that is related to the query
    - content: The content of the search result, that is related to the query; or summarized content of the page
    - links: The useful links in the search result, that is related to the query, if there are no useful links, you should return nothing

# Example Output:
## report
<short summary of the page content>
### Title
<title>

### Content
<content>

### Possible Useful Links
<links 1>: <link 1 description>
<links 2>: <link 2 description>
...




# Inputs:
Now you know the task, please perform extraction with the following inputs:

### Query:
{query}

### Crawling Result:
{crawling_result}

"""
