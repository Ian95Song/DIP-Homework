/**
 * Copyright 2019/2020 Andreas Ley
 * Written for Digital Image Processing of TU Berlin
 * For internal use in the course only
 */


#ifndef TASKSCHEDULER_H
#define TASKSCHEDULER_H

#include <thread>
#include <condition_variable>
#include <mutex>
#include <functional>
#include <atomic>
#include <memory>
#include <list>
#include <vector>
#include <stdexcept>


class TaskScheduler;

class TaskGroup;

class Task
{
    public:
        typedef std::function<void(void)> TaskFunction;
        Task(TaskFunction taskFunction, TaskGroup *group);
        
        inline void execute() { m_taskFunction(); }
        inline TaskGroup *getGroup() { return m_group; }
    private:
        TaskFunction m_taskFunction;
        TaskGroup *m_group;
};

class TaskGroup
{
    public:
        TaskGroup();
        ~TaskGroup();
        inline bool finished() { return m_remainingUnfinishedTasks.load() == 0; }
        void add(Task::TaskFunction taskFunction);
        void waitFor();
    protected:
        std::atomic<unsigned> m_remainingUnfinishedTasks;
        std::list<Task> m_tasks;
        friend class TaskScheduler;
};

class TaskScheduler
{
    public:
        static void Init(unsigned helperThreads);

        static TaskScheduler &get() { if (m_mainInstance != NULL) return *m_mainInstance; throw std::runtime_error("TaskScheduler not initialized yet!"); }

        inline std::size_t getNumThreads() const { return m_helperThreads.size(); }

        ~TaskScheduler();
    private:
        TaskScheduler(unsigned helperThreads);

        void scheduleTask(Task *task);

        static std::unique_ptr<TaskScheduler> m_mainInstance;

        std::atomic<bool> m_shutdown;
        typedef std::list<Task*> TaskList;
        TaskList m_scheduledTasks;

        std::condition_variable m_wakeupCondition;
        std::condition_variable m_groupFinishedCondition;
        std::mutex m_taskListMutex;
        std::mutex m_taskGroupMutex;


        std::vector<std::thread> m_helperThreads;

        void helperThreadOperate();
        
        friend class TaskGroup;
};

template<class Functor>
void parallelFor(int min, int max, int blockSize, const Functor &functor) {
    TaskGroup group;
    for (int i = min; i < max; i += blockSize)
        group.add([i, &functor, &max, &blockSize]{
            for (int j = i; j < std::min(i + blockSize, max); j++)
                functor(j);
        });
    group.waitFor();
}


#endif // TASKSCHEDULER_H
