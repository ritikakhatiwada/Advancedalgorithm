"""
Task 1: Advanced Data Structures
ST5003CEM — Advanced Algorithms

Implements:
  - Binary Search Tree (BST)         — insert, search, delete
  - AVL Tree (self-balancing)        — insert, search, delete
  - Min-Heap                         — insert, extract_min, peek
  - Hash Table (open addressing)     — insert, search, delete

Benchmarks all structures at n = 100, 1,000, 10,000 nodes.
"""
import time
import random
import sys
sys.setrecursionlimit(50000)

# ══════════════════════════════════════════════════════════════════════
# 1. BINARY SEARCH TREE (BST)
# ══════════════════════════════════════════════════════════════════════
class BSTNode:
    def __init__(self, key, data):
        self.key   = key
        self.data  = data
        self.left  = None
        self.right = None

class BST:
    """
    Binary Search Tree — stores city records keyed by city name.
    Average: O(log n) insert/search/delete
    Worst:   O(n)     when input is sorted (degenerates to linked list)
    Space:   O(n)
    """
    def __init__(self):
        self.root = None

    # ── Insert (iterative to avoid recursion limit) ────────────────
    def insert(self, key, data):
        if self.root is None:
            self.root = BSTNode(key, data)
            return
        node = self.root
        while True:
            if key < node.key:
                if node.left is None:
                    node.left = BSTNode(key, data); return
                node = node.left
            elif key > node.key:
                if node.right is None:
                    node.right = BSTNode(key, data); return
                node = node.right
            else:
                node.data = data; return  # update duplicate

    # ── Search (iterative) ─────────────────────────────────────────
    def search(self, key):
        node = self.root
        while node:
            if key == node.key: return node.data
            node = node.left if key < node.key else node.right
        return None

    # ── Delete (recursive) ────────────────────────────────────────
    def delete(self, key):
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        if node is None:
            return None
        if key < node.key:
            node.left  = self._delete(node.left,  key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            # Node found
            if node.left is None:  return node.right
            if node.right is None: return node.left
            # Two children: replace with inorder successor
            succ = node.right
            while succ.left:
                succ = succ.left
            node.key, node.data = succ.key, succ.data
            node.right = self._delete(node.right, succ.key)
        return node

    # ── Inorder traversal ─────────────────────────────────────────
    def inorder(self):
        result = []
        def _traverse(n):
            if n:
                _traverse(n.left)
                result.append(n.key)
                _traverse(n.right)
        _traverse(self.root)
        return result


# ══════════════════════════════════════════════════════════════════════
# 2. AVL TREE (Self-Balancing BST)
# ══════════════════════════════════════════════════════════════════════
class AVLNode:
    def __init__(self, key, data):
        self.key    = key
        self.data   = data
        self.left   = None
        self.right  = None
        self.height = 1

class AVLTree:
    """
    AVL Tree — guarantees O(log n) for all operations by keeping
    balance factor in {-1, 0, 1} via rotations after every insert/delete.
    Worst: O(log n) — guaranteed, unlike plain BST.
    Space: O(n)
    """
    def __init__(self):
        self.root = None

    def _h(self, n):
        return n.height if n else 0

    def _bf(self, n):
        return self._h(n.left) - self._h(n.right) if n else 0

    def _update_height(self, n):
        n.height = 1 + max(self._h(n.left), self._h(n.right))

    def _right_rotate(self, z):
        y, T3 = z.left, z.left.right
        y.right = z; z.left = T3
        self._update_height(z); self._update_height(y)
        return y

    def _left_rotate(self, z):
        y, T2 = z.right, z.right.left
        y.left = z; z.right = T2
        self._update_height(z); self._update_height(y)
        return y

    def _balance(self, node, key=None):
        self._update_height(node)
        bf = self._bf(node)
        # LL
        if bf > 1 and (key is None or key < node.left.key):
            return self._right_rotate(node)
        # RR
        if bf < -1 and (key is None or key > node.right.key):
            return self._left_rotate(node)
        # LR
        if bf > 1 and (key is None or key > node.left.key):
            node.left = self._left_rotate(node.left)
            return self._right_rotate(node)
        # RL
        if bf < -1 and (key is None or key < node.right.key):
            node.right = self._right_rotate(node.right)
            return self._left_rotate(node)
        return node

    # ── Insert ────────────────────────────────────────────────────
    def insert(self, key, data):
        self.root = self._insert(self.root, key, data)

    def _insert(self, node, key, data):
        if not node: return AVLNode(key, data)
        if   key < node.key: node.left  = self._insert(node.left,  key, data)
        elif key > node.key: node.right = self._insert(node.right, key, data)
        else: node.data = data; return node
        return self._balance(node, key)

    # ── Search ────────────────────────────────────────────────────
    def search(self, key):
        n = self.root
        while n:
            if key == n.key: return n.data
            n = n.left if key < n.key else n.right
        return None

    # ── Delete ────────────────────────────────────────────────────
    def delete(self, key):
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        if not node: return None
        if   key < node.key: node.left  = self._delete(node.left,  key)
        elif key > node.key: node.right = self._delete(node.right, key)
        else:
            if not node.left:  return node.right
            if not node.right: return node.left
            succ = node.right
            while succ.left: succ = succ.left
            node.key, node.data = succ.key, succ.data
            node.right = self._delete(node.right, succ.key)
        return self._balance(node)

    def inorder(self):
        result = []
        def _t(n):
            if n: _t(n.left); result.append(n.key); _t(n.right)
        _t(self.root)
        return result


# ══════════════════════════════════════════════════════════════════════
# 3. MIN-HEAP (Priority Queue)
# ══════════════════════════════════════════════════════════════════════
class MinHeap:
    """
    Min-Heap — array-based binary heap.
    insert:      O(log n)  — heapify up
    extract_min: O(log n)  — heapify down
    peek:        O(1)      — always index 0
    search:      O(n)      — not optimised for lookup
    Space:       O(n)
    """
    def __init__(self):
        self.heap = []   # list of (priority, data)

    def insert(self, priority, data):
        self.heap.append((priority, data))
        self._sift_up(len(self.heap) - 1)

    def _sift_up(self, i):
        while i > 0:
            p = (i - 1) // 2
            if self.heap[i][0] < self.heap[p][0]:
                self.heap[i], self.heap[p] = self.heap[p], self.heap[i]
                i = p
            else:
                break

    def extract_min(self):
        if not self.heap: return None
        min_val = self.heap[0]
        if len(self.heap) == 1:
            self.heap.pop()
        else:
            self.heap[0] = self.heap.pop()
            self._sift_down(0)
        return min_val

    def _sift_down(self, i):
        n = len(self.heap)
        while True:
            smallest = i
            l, r = 2*i+1, 2*i+2
            if l < n and self.heap[l][0] < self.heap[smallest][0]: smallest = l
            if r < n and self.heap[r][0] < self.heap[smallest][0]: smallest = r
            if smallest == i: break
            self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
            i = smallest

    def peek(self):
        return self.heap[0] if self.heap else None

    def __len__(self):
        return len(self.heap)


# ══════════════════════════════════════════════════════════════════════
# 4. HASH TABLE (Open Addressing — Linear Probing)
# ══════════════════════════════════════════════════════════════════════
class HashTable:
    """
    Hash Table with linear probing collision resolution.
    insert/search: O(1) average, O(n) worst case (high load factor)
    delete:        O(1) average — uses tombstone markers
    Maintains load factor < 0.7 via dynamic resizing (doubles capacity).
    Space: O(n)
    """
    _DELETED = object()   # Tombstone sentinel for deleted slots

    def __init__(self, capacity=128):
        self.capacity = capacity
        self.size     = 0
        self.table    = [None] * capacity

    def _hash(self, key):
        """Polynomial rolling hash."""
        h = 0
        for ch in key:
            h = (h * 31 + ord(ch)) % self.capacity
        return h

    def insert(self, key, data):
        if self.size / self.capacity > 0.7:
            self._resize()
        idx = self._hash(key)
        first_deleted = None
        while self.table[idx] is not None:
            if self.table[idx] is self._DELETED:
                if first_deleted is None:
                    first_deleted = idx
            elif self.table[idx][0] == key:
                self.table[idx] = (key, data)   # update
                return
            idx = (idx + 1) % self.capacity
        # Insert at first deleted slot or empty slot
        target = first_deleted if first_deleted is not None else idx
        if self.table[target] is None:
            self.size += 1
        self.table[target] = (key, data)

    def search(self, key):
        idx = self._hash(key)
        while self.table[idx] is not None:
            if self.table[idx] is not self._DELETED and self.table[idx][0] == key:
                return self.table[idx][1]
            idx = (idx + 1) % self.capacity
        return None

    def delete(self, key):
        """Delete using tombstone — avoids breaking probe chains."""
        idx = self._hash(key)
        while self.table[idx] is not None:
            if self.table[idx] is not self._DELETED and self.table[idx][0] == key:
                self.table[idx] = self._DELETED
                self.size -= 1
                return True
            idx = (idx + 1) % self.capacity
        return False   # Key not found

    def _resize(self):
        old = self.table
        self.capacity *= 2
        self.table = [None] * self.capacity
        self.size   = 0
        for item in old:
            if item and item is not self._DELETED:
                self.insert(item[0], item[1])


# ══════════════════════════════════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════════════════════════════════
def benchmark():
    sizes = [100, 1000, 10000]

    print("\n" + "="*70)
    print(f"{'Task 1: Data Structures — Insertion Benchmark':^70}")
    print("="*70)
    print(f"{'n':<8} {'BST Ins':>11} {'AVL Ins':>11} {'Heap Ins':>11} {'HT Ins':>11}")
    print("-"*55)

    for n in sizes:
        cities = [
            (f"City_{i:05d}",
             (round(random.uniform(-90, 90), 4),
              round(random.uniform(-180, 180), 4)),
             random.randint(1000, 10_000_000),
             round(random.uniform(0, 5000), 2))
            for i in range(n)
        ]
        random.shuffle(cities)   # random order avoids BST worst-case

        keys = [c[0] for c in cities]

        # BST insert
        bst = BST()
        t = time.perf_counter()
        for c in cities: bst.insert(c[0], c[1:])
        t_bst_ins = (time.perf_counter() - t) * 1000

        # AVL insert
        avl = AVLTree()
        t = time.perf_counter()
        for c in cities: avl.insert(c[0], c[1:])
        t_avl_ins = (time.perf_counter() - t) * 1000

        # Min-Heap insert
        heap = MinHeap()
        t = time.perf_counter()
        for c in cities: heap.insert(c[3], c[0])
        t_heap_ins = (time.perf_counter() - t) * 1000

        # Hash Table insert
        ht = HashTable()
        t = time.perf_counter()
        for c in cities: ht.insert(c[0], c[1:])
        t_ht_ins = (time.perf_counter() - t) * 1000

        print(f"{n:<8} {t_bst_ins:>10.2f}ms {t_avl_ins:>10.2f}ms "
              f"{t_heap_ins:>10.2f}ms {t_ht_ins:>10.2f}ms")

    print("\n" + "="*70)
    print(f"{'Search Benchmark (200 queries)':^70}")
    print("="*70)
    print(f"{'n':<8} {'BST Srch':>11} {'AVL Srch':>11} {'HT Srch':>11}")
    print("-"*44)

    for n in sizes:
        cities = [(f"City_{i:05d}", i * 10.5) for i in range(n)]
        random.shuffle(cities)
        sample = [cities[i][0] for i in range(min(200, n))]

        bst = BST(); avl = AVLTree(); ht = HashTable()
        for c in cities:
            bst.insert(c[0], c[1])
            avl.insert(c[0], c[1])
            ht.insert(c[0], c[1])

        t = time.perf_counter()
        for k in sample: bst.search(k)
        t_b = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        for k in sample: avl.search(k)
        t_a = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        for k in sample: ht.search(k)
        t_h = (time.perf_counter() - t) * 1000

        print(f"{n:<8} {t_b:>10.2f}ms {t_a:>10.2f}ms {t_h:>10.2f}ms")

    print("\n" + "="*70)
    print(f"{'Deletion Benchmark (100 deletes)':^70}")
    print("="*70)
    print(f"{'n':<8} {'BST Del':>11} {'AVL Del':>11} {'HT Del':>11}")
    print("-"*44)

    for n in sizes:
        cities = [(f"City_{i:05d}", i) for i in range(n)]
        random.shuffle(cities)
        to_delete = [cities[i][0] for i in range(min(100, n))]

        bst = BST(); avl = AVLTree(); ht = HashTable()
        for c in cities:
            bst.insert(c[0], c[1])
            avl.insert(c[0], c[1])
            ht.insert(c[0], c[1])

        t = time.perf_counter()
        for k in to_delete: bst.delete(k)
        t_b = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        for k in to_delete: avl.delete(k)
        t_a = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        for k in to_delete: ht.delete(k)
        t_h = (time.perf_counter() - t) * 1000

        print(f"{n:<8} {t_b:>10.2f}ms {t_a:>10.2f}ms {t_h:>10.2f}ms")

    print("\n" + "="*50)
    print("Min-Heap Priority Queue Demo")
    print("="*50)
    heap = MinHeap()
    cities_dist = [("Pokhara", 200), ("Chitwan", 150), ("Bhaktapur", 13),
                   ("Lumbini", 400), ("Biratnagar", 350)]
    print("Inserting cities with distances from Kathmandu:")
    for name, dist in cities_dist:
        heap.insert(dist, name)
        print(f"  inserted {name} (dist={dist}km)  heap size={len(heap)}")
    print("\nExtracting nearest cities (ascending order):")
    while len(heap) > 0:
        dist, city = heap.extract_min()
        print(f"  {city:<14} — {dist} km")

    print("\n" + "="*50)
    print("Hash Table Tombstone Delete Demo")
    print("="*50)
    ht2 = HashTable(capacity=16)
    for name in ["Alice", "Bob", "Charlie", "Dave"]:
        ht2.insert(name, name + "@email.com")
    print(f"  Search 'Bob' before delete : {ht2.search('Bob')}")
    ht2.delete("Bob")
    print(f"  Search 'Bob' after delete  : {ht2.search('Bob')}  (None = correct)")
    print(f"  Search 'Charlie' (intact)  : {ht2.search('Charlie')}")

    print("\n" + "="*50)
    print("BST vs AVL — Worst Case (Sorted Input)")
    print("="*50)
    n = 500
    sorted_keys = [f"City_{i:05d}" for i in range(n)]
    bst_sorted = BST(); avl_sorted = AVLTree()
    t = time.perf_counter()
    for k in sorted_keys: bst_sorted.insert(k, k)
    t_bst = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    for k in sorted_keys: avl_sorted.insert(k, k)
    t_avl = (time.perf_counter() - t) * 1000
    print(f"  Sorted insert n={n}:")
    print(f"  BST: {t_bst:.2f}ms  (degenerates — O(n) per op = O(n²) total)")
    print(f"  AVL: {t_avl:.2f}ms  (balanced — O(log n) per op = O(n log n) total)")
    print(f"  AVL speedup on sorted data: {t_bst/t_avl:.1f}x faster")


if __name__ == "__main__":
    benchmark()
