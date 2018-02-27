#![deny(warnings)]
#![cfg_attr(feature = "cargo-clippy", allow(float_cmp))]
extern crate mpi;

use mpi::traits::*;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();

    let rank = world.rank();
    let size = world.size();

    // send vec![rank; size] left and right
    mpi::request::scope(|scope| {
        let left_requests = if rank > 0 {
            let left_process = world.process_at_rank(rank - 1);

            Some((
                left_process.immediate_send(scope, vec![rank; size as usize]),
                left_process.immediate_receive_into(scope, vec![0; size as usize]),
            ))
        } else {
            None
        };

        let right_requests = if rank < size - 1 {
            let right_process = world.process_at_rank(rank + 1);

            Some((
                right_process.immediate_send(scope, vec![rank; size as usize]),
                right_process.immediate_receive_into(scope, vec![0; size as usize]),
            ))
        } else {
            None
        };

        let left_send = left_requests.map(|(left_send, left_recv)| {
            let (recv_data, _) = left_recv.wait_recv();

            assert_eq!(vec![rank - 1; size as usize], recv_data);

            left_send
        });

        let right_send = right_requests.map(|(right_send, right_recv)| {
            let (recv_data, _) = right_recv.wait_recv();

            assert_eq!(vec![rank + 1; size as usize], recv_data);

            right_send
        });

        if let Some(sreq) = left_send {
            sreq.wait();
        }

        if let Some(sreq) = right_send {
            sreq.wait();
        }
    });
}
