# Dspy fun
Playing with Dspy to automate prompt optimization
1. **AG news dataset** - 
The plan is to use the AG news dataset to classify news articles into 4 categories: World, Sports, Business, and Sci/Tech. We will use Dspy to optimize prompts for a language model to improve classification accuracy.
We use Dspy to create a prompt template and optimize it.

Base prompt before optimization:
```
"Your input fields are:
1. `text` (str): News article text.
Your output fields are:
1. `label` (Literal['World', 'Sports', 'Business', 'Sci/Tech']): Category of the news article.
2. `confidence` (float):
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## text ## ]]
{text}

[[ ## label ## ]]

{label}        # note: the value you produce must exactly match (no extra characters) one of: World; Sports; Business; Sci/Tech

[[ ## confidence ## ]]
{confidence}        # note: the value you produce must be a single float value

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Classify news articles into categories."
```

Accuracy before optimization: 44%

After running Dspy optimization using `MIPROv2`, we obtained the following optimized prompt:
```
Your input fields are:
1. `text` (str):
Your output fields are:
1. `label` (Literal['Sports', 'Sci/Tech', 'Business', 'World']): 
2. `confidence` (float):
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## text ## ]]
{text}

[[ ## label ## ]]
{label}        # note: the value you produce must exactly match (no extra characters) one of: Sports; Sci/Tech; Business; World

[[ ## confidence ## ]]
{confidence}        # note: the value you produce must be a single float value

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        Prompt the Language Model to analyze text and predict a label and confidence value based on the text's context and domain.
```
Accuracy after optimization: 67%