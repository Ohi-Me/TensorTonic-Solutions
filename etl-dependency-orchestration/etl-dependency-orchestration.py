def schedule_pipeline(tasks: list, resource_budget: int) -> list:
    n=len(tasks)
    task_map={task["name"]:task for task in tasks}
    completed=set()
    started=set()
    running=[]
    ans=[]
    time=0
    used=0

    while len(completed)<n:
        i=0
        while i<len(running):
            end_time,name=running[i]

            if end_time==time:
                used-=task_map[name]["resources"]
                completed.add(name)
                running.pop(i)
            else:
                i+=1

        ready=[]

        for task in tasks:
            if task["name"] in started:
                continue

            if all(dep in completed for dep in task["depends_on"]):
                ready.append(task)

        ready.sort(key=lambda x:x["name"])

        for task in ready:
            if used+task["resources"]<=resource_budget:
                name=task["name"]
                started.add(name)
                used+=task["resources"]
                running.append((time+task["duration"],name))
                ans.append({
                    "task_name":name,
                    "start_time":time
                })

        if running:
            time=min(end_time for end_time,name in running)

    ans.sort(key=lambda x:(x["start_time"],x["task_name"]))
    return ans