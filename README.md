# zenoh_fl

## Installing

To run the scenarios locally please install the requirements

 `pip install -r requirements.txt`

## Running

To run locally you only need the following commands

`mpirun -np 4 python mpi_decentralized_assync.py -d dataset/one_hot_encoding/`

`mpirun -np 4 python mpi_decentralized_sync.py -d dataset/one_hot_encoding/`

`mpirun -np 4 python mpi_centralized_sync.py -d dataset/one_hot_encoding/`

`mpirun -np 4 python mpi_centralized_assync.py -d dataset/one_hot_encoding/`

If you want to run the single_host setting you can run

`python single_host.py`

## Results


## Authors

* **Rafael Teixeira** - [rgtzths](https://github.com/rgtzths)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
