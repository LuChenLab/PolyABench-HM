# Example input/output

- `example.fa`: 6 Human 240-nt sequences sampled from the Human validation set.
  - 3 positive poly(A) sites (headers ending in `:1`)
  - 3 negative sequences (headers ending in `:0`)
- `example_predictions.txt`: output of running the best ensemble (`-m ensemble`) on `example.fa`.

Run the example:

```bash
apa-polya-predict -i examples/example.fa -s Human -m ensemble -o predictions.txt
```
