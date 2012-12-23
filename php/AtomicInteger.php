<?php

require_once __DIR__ . '/Mutex.php';

class ShmInteger {
    protected $resource = null;
    protected $shm_key = null;
    private function __construct($key) {
        $this->shm_key = $key;
        $this->resource = shm_attach($this->shm_key);
    }
    public static function initialize($key) {
        $i = self::createInstance(0, $key);
        $i->destroy();
    }
    public static function createInstance($init, $key) {
        $instance = new ShmInteger($key);
        $instance->set($init);
        return $instance;
    }
    public function destroy() {
        shm_remove($this->resource);
        $this->resource = false;
    }
    public function get() {
        return shm_get_var($this->resource, 0);
    }
    public function set($v) {
        shm_put_var($this->resource, 0, $v);
    }
    public function __destruct() {
        if ($this->resource !== false) shm_detach($this->resource);
    }
}

class AtomicInteger extends Mutex {
    protected $integer = null;
    protected $name = null;
    public function __construct($init = 0, $path, $key) {
        parent::__construct($path, $key);
        $this->name = "$path/$key";
        ShmInteger::initialize($this->shm_key);
        $this->integer = ShmInteger::createInstance($init, $this->shm_key);
    }
    public function getName() {
        return $this->name;
    }
    public function decrement() {
        sem_acquire($this->mutex);
        $old = $this->integer->get();
        $this->integer->set($old - 1);
        sem_release($this->mutex);
    }
    public function decrementAndGet() {
        $value = 0;
        sem_acquire($this->mutex);
        $old = $this->integer->get();
        $this->integer->set($old - 1);
        $value = $this->integer->get();
        sem_release($this->mutex);
        return $value;
    }
    public function get() {
        $value = 0;
        sem_acquire($this->mutex);
        $value = $this->integer->get();
        sem_release($this->mutex);
        return $value;
    }
    public function getAndIncrement() {
        $value = 0;
        sem_acquire($this->mutex);
        $value = $this->integer->get();
        $this->integer->set($value + 1);
        sem_release($this->mutex);
        return $value;
    }
    public function getAndDecrement() {
        $value = 0;
        sem_acquire($this->mutex);
        $value = $this->integer->get();
        $this->integer->set($value - 1);
        sem_release($this->mutex);
        return $value;
    }
    public function increment() {
        sem_acquire($this->mutex);
        $old = $this->integer->get();
        $this->integer->set($old + 1);
        sem_release($this->mutex);
    }
    public function incrementAndGet() {
        $value = 0;
        sem_acquire($this->mutex);
        $old = $this->integer->get();
        $this->integer->set($old + 1);
        $value = $this->integer->get();
        sem_release($this->mutex);
        return $value;
    }
    public function set($init = 0) {
        sem_acquire($this->mutex);
        $this->integer->set($init);
        sem_release($this->mutex);
    }
}

?>
