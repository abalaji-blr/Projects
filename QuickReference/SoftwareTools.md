# Software Tools
## Popular IDEs 
* Cursor - Integration with multiple AI models
* VS Code - Download Gemini Code Assist is **FREE** for developers.

## Version Control
* Git - Often Used commands
  ```
  * git init                      // init the repo
  * git add .                     // add all files to track
  * git commit -m "my comment" files
  * git log
  * git branch branchName        // creates new branch
  * git checkout branchName      // make the branch active
  ```
    
* Learn [Git branching using Game](https://learngitbranching.js.org/)

## Python Package Manager

## Debug - Python
* Python
  ```
  import code
  code.interact()

  >> exit() or
  >> ctrl+D  -- continue execute the program
  ```
* Python debugger (pdb)
  ```
  from pdb import set_trace

  set_trace()
  
  pdb > help
  > n  // next
  > c  // continue
  > l  -- list
  > q -- quit
  > b # -- break point line-num
  > cl  -- clear all breakpoints
  > s -- step
  > p -- print
  ```
* Dotenv
  
  pip install dotenv
  
  .env - houses all critical keys.
  
  ```
  from dotenv import load_dotenv
  import os

  api_key = os.getenv("API_KEY")
  ```
  
