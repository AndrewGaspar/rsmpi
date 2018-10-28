#![deny(warnings)]
extern crate mpi;

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();

    let comm = universe.world();

    if comm.size() < 4 {
        return;
    }

    let cart_comm = {
        let dims = [2, 2];
        let periodic = [false, true];
        let reorder = true;
        if let Some(cart_comm) = comm.create_cartesian_communicator(&dims, &periodic, reorder) {
            cart_comm
        } else {
            assert!(comm.rank() >= 4);
            return;
        }
    };

    assert_eq!(2, cart_comm.num_dimensions());

    let mpi::topology::CartesianLayout {
        dims,
        periods,
        coords,
    } = cart_comm.get_layout();

    assert_eq!([2 as mpi::Count, 2], &dims[..]);
    assert_eq!([false, true], &periods[..]);

    assert!(0 <= coords[0] && coords[0] < 2);
    assert!(0 <= coords[1] && coords[1] < 2);
}
