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

    let xrank = coords[0];
    let yrank = coords[1];

    assert!(0 <= xrank && xrank < 2);
    assert!(0 <= yrank && yrank < 2);

    let xcomm = cart_comm.subgroup(&[true, false]);
    let ycomm = cart_comm.subgroup(&[false, true]);

    assert_eq!(2, xcomm.size());
    assert_eq!(xrank, xcomm.rank());

    assert_eq!(2, ycomm.size());
    assert_eq!(yrank, ycomm.rank());
}
