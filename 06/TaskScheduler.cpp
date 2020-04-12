/**
 * Copyright 2019/2020 Andreas Ley
 * Written for Digital Image Processing of TU Berlin
 * For internal use in the course only
 */


#include "TaskScheduler.h"

Task::Task(TaskFunction taskFunction, TaskGroup *group) : m_taskFunction(std::move(taskFunction)), m_group(group) {
}


TaskGroup::TaskGroup()
{
    m_remainingUnfinishedTasks.store(0);
}

TaskGroup::~TaskGroup()
{
    waitFor();
}

void TaskGroup::add(Task::TaskFunction taskFunction)
{
    m_tasks.push_back(Task(std::move(taskFunction), this));
    m_remainingUnfinishedTasks++;
    TaskScheduler::get().scheduleTask(&m_tasks.back());
}

void TaskGroup::waitFor()
{
    std::unique_lock<std::mutex> lock(TaskScheduler::get().m_taskGroupMutex);
    while (m_remainingUnfinishedTasks.load() > 0) {
        TaskScheduler::get().m_groupFinishedCondition.wait(lock);
    }
}

void TaskScheduler::Init(unsigned helperThreads)
{
    if (m_mainInstance != nullptr)
        throw std::runtime_error("Already initialized!");

    m_mainInstance.reset(new TaskScheduler(helperThreads));
}

std::unique_ptr<TaskScheduler> TaskScheduler::m_mainInstance;


TaskScheduler::TaskScheduler(unsigned helperThreads)
{
    m_shutdown.store(false);

    m_helperThreads.resize(helperThreads);
    for (unsigned i = 0; i < helperThreads; i++)
        m_helperThreads[i] = std::thread(std::bind(&TaskScheduler::helperThreadOperate, this));
}

TaskScheduler::~TaskScheduler()
{
    {
        std::lock_guard<std::mutex> lock(m_taskListMutex);
        m_shutdown.store(true);
        m_wakeupCondition.notify_all();
    }

    for (unsigned i = 0; i < m_helperThreads.size(); i++)
        m_helperThreads[i].join();
}


void TaskScheduler::scheduleTask(Task *task)
{
    std::lock_guard<std::mutex> lock(m_taskListMutex);
    m_scheduledTasks.push_back(task);
    m_wakeupCondition.notify_one();
}


void TaskScheduler::helperThreadOperate()
{
    while (true) {
        Task *task = nullptr;
        {
            std::unique_lock<std::mutex> lock(m_taskListMutex);
            while (m_scheduledTasks.empty()) {
                if (m_shutdown.load())
                    return;
                m_wakeupCondition.wait(lock);
            }
            task = m_scheduledTasks.front();
            m_scheduledTasks.pop_front();
        }
        task->execute();
        {
            std::unique_lock<std::mutex> lock(m_taskGroupMutex);
            task->getGroup()->m_remainingUnfinishedTasks--;
            if (task->getGroup()->m_remainingUnfinishedTasks.load() == 0)
                m_groupFinishedCondition.notify_all();
        }
    }
}
