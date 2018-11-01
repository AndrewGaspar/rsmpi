#![deny(warnings)]
#![cfg_attr(feature = "cargo-clippy", allow(float_cmp))]
extern crate mpi;

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let rank = world.rank();
    let size = world.size();

    let my_data = vec![rank; size as usize];

    // send vec![rank; size] left and right
    let left_requests = if rank > 0 {
        let left_process = world.process_at_rank(rank - 1);

        Some((
            left_process.immediate_send(&my_data[..]),
            left_process.immediate_receive_into(vec![0; size as usize]),
        ))
    } else {
        None
    };

    let right_requests = if rank < size - 1 {
        let right_process = world.process_at_rank(rank + 1);

        Some((
            right_process.immediate_send(&my_data[..]),
            right_process.immediate_receive_into(vec![0; size as usize]),
        ))
    } else {
        None
    };

    let left_send = left_requests.map(|(left_send, left_recv)| {
<<<<<<< Updated upstream
        assert_eq!(
            vec![rank - 1; size as usize],
            left_recv.wait_data_without_status()
        );
=======
        let recv_data = left_recv.wait_data();

        assert_eq!(vec![rank - 1; size as usize], recv_data);
>>>>>>> Stashed changes

        left_send
    });

    let right_send = right_requests.map(|(right_send, right_recv)| {
<<<<<<< Updated upstream
        assert_eq!(
            vec![rank + 1; size as usize],
            right_recv.wait_data_without_status()
        );
=======
        let recv_data = right_recv.wait_data();

        assert_eq!(vec![rank + 1; size as usize], recv_data);
>>>>>>> Stashed changes

        right_send
    });

    if let Some(sreq) = left_send {
<<<<<<< Updated upstream
        assert_eq!(&my_data[..], sreq.wait_data_without_status());
    }

    if let Some(sreq) = right_send {
        assert_eq!(&my_data[..], sreq.wait_data_without_status());
=======
        let send_data = sreq.wait_data();
        assert_eq!(&my_data[..], send_data);
    }

    if let Some(sreq) = right_send {
        let send_data = sreq.wait_data();
        assert_eq!(&my_data[..], send_data);
>>>>>>> Stashed changes
    }
}
