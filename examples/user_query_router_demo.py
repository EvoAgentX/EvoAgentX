# demo file for UserQueryRouter

from evoagentx.app.components.user_query_router import UserQueryRouter


test_queries = {
    "normal_queries": [
        "I want to add a new node at the end of the workflow to print all middle variables in the workflow",
        "I want to add a button to the frontend to show the current time",
        "I want to modify the prompt of the agents in the last two workflow's nodes, give more details about the context to make the workflow work better",
        "I want to add a new node at the end of the workflow to print all middle variables in the workflow and make the Data Collection Node more performant",
    ],
    "unclear_queries": [
        "I want to add a new node at the end of the workflow",
    ]

}

if __name__ == "__main__":
    user_query_router = UserQueryRouter()
    results = user_query_router.route_query(
        user_query=test_queries["normal_queries"][3],
        workflow_context={}
    )
    print(results)