<?php

class Mutex {
    protected $mutex = null;
    protected $shm_key = 0;
    public function __construct($path, $key) {
        $this->shm_key = ftok($path, $key);
        $this->mutex = sem_get($this->shm_key);
    }
    public function lockThenRun($cb) {
        sem_acquire($this->mutex);
        $results = null;
        try {
            $results = $cb();
        } catch (Exception $e) {
            printf("Caught exception executing callback: %s\n", $e->getMessage());
        }
        sem_release($this->mutex);
        return $results;
    }
    public function println($message) {
        $this->lockThenRun(function() use($message) {
            printf("%s\n", $message);
        });
    }
}

?>
