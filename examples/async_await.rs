use mpi::topology::Rank;
use mpi::traits::*;

use futures::future::join_all;

fn main() {
    let universe = mpi::initialize().unwrap();

    let comm = universe.world();

    async_std::task::block_on(async {
        let mut left_recvs: Vec<Rank> = vec![0; comm.size() as usize];
        let mut right_recvs: Vec<Rank> = vec![0; comm.size() as usize];

        let sends: Vec<Rank> = (0..comm.size()).collect();

        {
            let left_recvs = &mut left_recvs;
            let right_recvs = &mut right_recvs;

            let sends = &sends;

            mpi::request::async_scope(|scope| {
                async move {
                    let recv_requests = left_recvs.iter_mut().zip(right_recvs.iter_mut()).flat_map(
                        |(left, right)| {
                            let lrank = if comm.rank() == 0 {
                                comm.size() - 1
                            } else {
                                comm.rank() - 1
                            };
                            let rrank = if comm.rank() == comm.size() - 1 {
                                0
                            } else {
                                comm.rank() + 1
                            };

                            vec![
                                comm.process_at_rank(lrank)
                                    .immediate_receive_into(scope.clone(), left)
                                    .into_future(),
                                comm.process_at_rank(rrank)
                                    .immediate_receive_into(scope.clone(), right)
                                    .into_future(),
                            ]
                        },
                    );

                    let send_requests = sends.iter().flat_map(|send| {
                        let lrank = if comm.rank() == 0 {
                            comm.size() - 1
                        } else {
                            comm.rank() - 1
                        };
                        let rrank = if comm.rank() == comm.size() - 1 {
                            0
                        } else {
                            comm.rank() + 1
                        };

                        vec![
                            comm.process_at_rank(lrank)
                                .immediate_send(scope.clone(), send)
                                .into_future(),
                            comm.process_at_rank(rrank)
                                .immediate_send(scope.clone(), send)
                                .into_future(),
                        ]
                    });

                    join_all(send_requests.chain(recv_requests)).await;
                }
            })
            .await;
        }

        println!("{}-left: {:?}", comm.rank(), &left_recvs);
        println!("{}-right: {:?}", comm.rank(), &right_recvs);
    });
}
