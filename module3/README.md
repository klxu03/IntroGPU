assignment.cu contains the CUDA code for the assignment.

Run make to build the assignment. Can execute the program with ./assignment.exe <number of threads> <number of blocks>
Note the number of threads and blocks can be different for each run and are optional arguments with default values of 1048576 and 256 respectively.

The program will output the time taken for each version of the mod operation and the results of the operations.
The program will also output some sample results from the operation that show the correctness of each operation type.

# Types of Mod Operations in Our CUDA Code

Below are **three different mod approaches** we used. Each one tackles the zero‐divisor challenge in a different way.

---

## 1) Branching Mod

- Checks if the second array’s value is zero. If so, it does `% 1`, otherwise it does `% arr2[idx]`.  

---

## 2) No-Branch Mod (Arithmetic Trick)

- Instead of using an `if` to check for zero, it simply adds 1 if `arr2[idx]` is zero. That way, there is never a zero divisor.  

---

## 3) Partition-Based (Specialized) Mod

- Partitions or reorganizes data so each kernel call deals with only one known divisor (for example, all threads do `% 2` or all do `% 3`).  
- For powers of two, it can replace `% 2` with bitwise AND, etc.  