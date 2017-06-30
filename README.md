# keras-cntk-benchmark

Repository to benchmark the performance of the CNTK backend on Keras vs. the performance of TensorFlow. This R Notebook is the complement to my blog post [Benchmarking CNTK on Keras: is it Better at Deep Learning than TensorFlow?](http://minimaxir.com/2017/06/keras-cntk/).

## Usage

The repository uses my [keras-cntk-docker](https://github.com/minimaxir/keras-cntk-docker) container, and assumes necessary dependices for that are installed.

`keras_cntk_benchmark.py` contains the benchmark script.

`/test_files` contains the test files.

`/logs` contains the performance output (CSV) for each test.

`/analysis` contains the R Notebook of the logs used to create the interactive data visualizations in the blog post. (You can view the R Notebook on the Web here)

## Maintainer

Max Woolf ([@minimaxir](http://minimaxir.com))

*Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use.*

## Credits

Test scripts (aside from `text_generator_keras.py`) sourced from the official Keras repository/contributors.

## License

MIT