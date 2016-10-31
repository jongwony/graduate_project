from multiprocessing import Process, Queue
import time

def f(q):
    print 'function in'
    time.sleep(7)
    q.put([42, None, 'hello'])
    print 'function out'

def r(q):
    print 'r function in'
    time.sleep(3)
    q.put([54, None, 'asf'])
    print 'r function out'

if __name__ == '__main__':
    q = Queue()
    p = Process(target=f, args=(q,))
    p2 = Process(target=r, args=(q,))
    p.start()
    p2.start()

    print 'process_start'
    p2.join()
    print 'process2_end'
    p.join()
    print 'process1_end'
