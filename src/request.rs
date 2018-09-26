//! Request objects for non-blocking operations
//!
//! Non-blocking operations such as `immediate_send()` return request objects that borrow any
//! buffers involved in the operation so as to ensure proper access restrictions. In order to
//! release the borrowed buffers from the request objects, a completion operation such as
//! [`wait()`](struct.Request.html#method.wait) or [`test()`](struct.Request.html#method.test) must
//! be used on the request object.
//!
//! **Note:** If the `Request` is dropped (as opposed to calling `wait` or `test` explicitly), the
//! program will panic.
//!
//! To enforce this rule, every request object must be registered to some pre-existing
//! [`Scope`](trait.Scope.html).  At the end of a `Scope`, all its remaining requests will be waited
//! for until completion.  Scopes can be created using either [`scope`](fn.scope.html) or
//! [`StaticScope`](struct.StaticScope.html).
//!
//! To handle request completion in an RAII style, a request can be wrapped in either
//! [`WaitGuard`](struct.WaitGuard.html) or [`CancelGuard`](struct.CancelGuard.html), which will
//! follow the respective policy for completing the operation.  When the guard is dropped, the
//! request will be automatically unregistered from its `Scope`.
//!
//! # Unfinished features
//!
//! - **3.7**: Nonblocking mode:
//!   - Completion, `MPI_Waitany()`, `MPI_Waitall()`, `MPI_Waitsome()`,
//!   `MPI_Testany()`, `MPI_Testall()`, `MPI_Testsome()`, `MPI_Request_get_status()`
//! - **3.8**:
//!   - Cancellation, `MPI_Test_cancelled()`

use std::cell::Cell;
use std::mem::{forget, replace, uninitialized};
use std::ptr::drop_in_place;
use std::marker::PhantomData;
use std::slice;

use conv::ConvUtil;

use ffi;
use ffi::{MPI_Request, MPI_Status};
use datatype::{ScopedBuffer, ScopedBufferMut};
use traits::*;
use point_to_point::Status;
use raw::traits::*;
use raw;

/// Request traits
pub mod traits {
    pub use super::{AsyncRequest, CollectRequests};
}

/// A request object for a non-blocking operation.
///
/// # Panics
///
/// Panics if the request object is dropped.  To prevent this, call `wait`, `wait_without_status`,
/// or `test`.  Alternatively, wrap the request inside a `WaitGuard` or `CancelGuard`.
///
/// # Examples
///
/// See `examples/immediate.rs`
///
/// # Standard section(s)
///
/// 3.7.1
pub trait AsyncRequest<Owned>: AsRaw<Raw = MPI_Request> + Sized {
    /// Unregister the request object from its scope and deconstruct it into its raw parts.
    ///
    /// This is unsafe because the request may outlive its associated buffers.
    unsafe fn into_raw(self) -> (MPI_Request, Owned);

    /// Wait for an operation to finish.
    ///
    /// Will block execution of the calling thread until the associated operation has finished.
    ///
    /// # Examples
    ///
    /// See `examples/immediate.rs`
    ///
    /// # Standard section(s)
    ///
    /// 3.7.3
    fn wait(self) -> Status {
        let mut status: MPI_Status = unsafe { uninitialized() };
        raw::wait(unsafe { &mut self.into_raw().0 }, Some(&mut status));
        Status::from_raw(status)
    }

    /// Waits for completion of the request and relinquishes ownership of its send and receive
    /// buffers.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.3
    fn wait_data(self) -> (Owned, Status) {
        let (mut request, data) = unsafe { self.into_raw() };

        let mut status: MPI_Status = unsafe { uninitialized() };
        raw::wait(&mut request, Some(&mut status));

        (data, Status::from_raw(status))
    }

    /// Wait for an operation to finish, but don’t bother retrieving the `Status` information.
    ///
    /// Will block execution of the calling thread until the associated operation has finished.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.3
    fn wait_without_status(self) {
        raw::wait(unsafe { &mut self.into_raw().0 }, None)
    }

    /// Wait for an operation to finish, but don’t bother retrieving the `Status` information.
    ///
    /// Will block execution of the calling thread until the associated operation has finished.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.3
    fn wait_data_without_status(self) -> Owned {
        let (mut request, data) = unsafe { self.into_raw() };
        raw::wait(&mut request, None);
        data
    }

    /// Test whether an operation has finished.
    ///
    /// If the operation has finished, `Status` is returned.  Otherwise returns the unfinished
    /// `Request`.
    ///
    /// # Examples
    ///
    /// See `examples/immediate.rs`
    ///
    /// # Standard section(s)
    ///
    /// 3.7.3
    fn test(self) -> Result<Status, Self> {
        match raw::test(&mut self.as_raw()) {
            Some(status) => {
                unsafe { self.into_raw() };
                Ok(Status::from_raw(status))
            }
            None => Err(self),
        }
    }

    /// Test whether an operation has finished.
    ///
    /// If the operation has finished, `Status` and the owned data is returned.  Otherwise returns
    /// the unfinished `Request`.
    ///
    /// # Examples
    ///
    /// See `examples/immediate.rs`
    ///
    /// # Standard section(s)
    ///
    /// 3.7.3
    fn test_data(self) -> Result<(Owned, Status), Self> {
        match raw::test(&mut self.as_raw()) {
            Some(status) => {
                let (_, data) = unsafe { self.into_raw() };
                Ok((data, Status::from_raw(status)))
            }
            None => Err(self),
        }
    }

    /// Initiate cancellation of the request.
    ///
    /// The MPI implementation is not guaranteed to fulfill this operation.  It may not even be
    /// valid for certain types of requests.  In the future, the MPI forum may [deprecate
    /// cancellation of sends][mpi26] entirely.
    ///
    /// [mpi26]: https://github.com/mpi-forum/mpi-issues/issues/26
    ///
    /// # Examples
    ///
    /// See `examples/immediate.rs`
    ///
    /// # Standard section(s)
    ///
    /// 3.8.4
    fn cancel(&self) {
        let mut request = self.as_raw();
        unsafe {
            ffi::MPI_Cancel(&mut request);
        }
    }
}

/// A request object for a non-blocking operation.
///
/// # Panics
///
/// Panics if the request object is dropped.  To prevent this, call `wait`, `wait_without_status`,
/// or `test`.  Alternatively, wrap the request inside a `WaitGuard` or `CancelGuard`.
///
/// # Examples
///
/// See `examples/immediate.rs`
///
/// # Standard section(s)
///
/// 3.7.1
#[must_use]
#[derive(Debug)]
pub struct Request<Owned = ()> {
    request: MPI_Request,
    data: Owned,
}

/// A `SendRequest` is a request that is sending data. It maintains either a const borrow or
/// ownership of the buffer being sent.
pub type SendRequest<S> = Request<S>;

/// A `RecvRequest` is a request that is receiving data. It maintains either a mutable borrow or
/// ownership of the buffer receiving the data.
pub type RecvRequest<R> = Request<R>;

/// A `SendRecvRequest` is a request that is sending and receiving data. It maintains a const
/// borrow or ownership of the buffer being sent, and a mutable borrow or ownership of the buffer
/// receiving the data.
pub type SendRecvRequest<S, R> = Request<(S, R)>;

unsafe impl<Owned> AsRaw for Request<Owned> {
    type Raw = MPI_Request;
    fn as_raw(&self) -> Self::Raw {
        self.request
    }
}

impl<Owned> Drop for Request<Owned> {
    fn drop(&mut self) {
        panic!("request was dropped without being completed");
    }
}

impl<Owned> Request<Owned> {
    /// Stops tracking the request's data buffers. The lifetime of the buffers must exceed the
    /// lifetime of the attached scope.
    // pub fn forget_data(self) -> Request
    // where
    //     S: 'a,
    //     R: 'a,
    // {
    //     unsafe {
    //         let (request, scope, _, _) = self.into_raw_data();
    //         Request::from_raw(request, scope)
    //     }
    // }

    /// Construct a request object from the raw MPI type and its associated data buffer.
    ///
    /// # Requirements
    ///
    /// - The request is a valid, active request.  It must not be `MPI_REQUEST_NULL`.
    /// - The request must not be persistent.
    /// - All buffers associated with the request must outlive `'a`.
    /// - The request must not be registered with the given scope.
    /// - If `data` is a reference, the referenced must live at least as long as the MPI Request.
    ///
    pub unsafe fn from_raw(request: MPI_Request, data: Owned) -> Self {
        debug_assert!(!request.is_handle_null());
        Self { request, data }
    }
}

impl<Owned> AsyncRequest<Owned> for Request<Owned> {
    /// Unregister the request object from its scope and deconstruct it into its raw parts.
    ///
    /// This is unsafe because the request may outlive its associated buffers.
    unsafe fn into_raw(mut self) -> (MPI_Request, Owned) {
        let request = replace(&mut self.request, uninitialized());
        let data = replace(&mut self.data, uninitialized());
        forget(self);
        (request, data)
    }
}

/// Collects an iterator of `Request` objects into a `RequestCollection` object
pub trait CollectRequests<Owned>: IntoIterator<Item = Request<Owned>> {
    /// Consumes and converts an iterator of `Requst` objects into a `RequestCollection` object.
    fn collect_requests(self) -> RequestCollection<Owned>;
}

impl<Owned, T> CollectRequests<Owned> for T
where
    T: IntoIterator<Item = Request<Owned>>,
{
    fn collect_requests(self) -> RequestCollection<Owned> {
        RequestCollection::from_request_iter(self)
    }
}

/// Result type for `RequestCollection::test_any`.
#[derive(Clone, Copy)]
pub enum TestAny<Owned> {
    /// Indicates that there are no active requests in the `requests` slice.
    NoneActive,
    /// Indicates that, while there are active requests in the `requests` slice, none of them were
    /// completed.
    NoneComplete,
    /// Indicates which request in the `requests` slice was completed.
    Completed(i32, Owned, Status),
}

/// Result type for `RequestCollection::test_any_without_status`.
#[derive(Clone, Copy)]
pub enum TestAnyWithoutStatus<Owned> {
    /// Indicates that there are no active requests in the `requests` slice.
    NoneActive,
    /// Indicates that, while there are active requests in the `requests` slice, none of them were
    /// completed.
    NoneComplete,
    /// Indicates which request in the `requests` slice was completed.
    Completed(i32, Owned),
}

/// A collection of request objects for a non-blocking operation registered with a `Scope` of
/// lifetime `'a`.
///
/// The `Scope` is needed to ensure that all buffers associated request will outlive the request
/// itself, even if the destructor of the request fails to run.
///
/// # Panics
///
/// Panics if the collection is dropped while it contains outstanding requests.
/// To prevent this, call `wait_all` or repeatedly call `wait_some`, `wait_any`, `test_any`,
/// `test_some`, or `test_all` until all requests are reported as complete.
///
/// # Examples
///
/// See `examples/immediate_wait_all.rs`
///
/// # Standard section(s)
///
/// 3.7.5
#[must_use]
#[derive(Debug)]
pub struct RequestCollection<Owned> {
    /// Tracks the number of request handles in `requests` are active.
    outstanding: usize,

    /// NOTE: Once Rust supports some sort of "null pointer optimization" for custom types, this
    /// could become essentially `Vec<Option<MPI_Request>>`.
    requests: Vec<MPI_Request>,

    /// `data[i]` contains the owned objects and buffers associated with `requests[i]`
    ///
    /// `data` is not used like a typical Vec - so as to not impose `Default` requirements
    /// on `S` and not have to wrap `S` in `Option`, we manage the uninitialized memory in
    /// accordance with the active state of the associated request. This also allows no memory to be
    /// allocated for `data` when `Owned = ()`.
    ///
    /// Therefore, `data.len()` will always be `0`.
    data: Vec<Owned>,
}

impl<Owned> Drop for RequestCollection<Owned> {
    fn drop(&mut self) {
        if self.outstanding != 0 {
            panic!("RequestCollection was dropped with outstanding requests not completed.");
        }
    }
}

impl<Owned> RequestCollection<Owned> {
    /// Constructs a `RequestCollection` from a Vec of `MPI_Request` handles and a scope object.
    /// `requests` are allowed to be null, but they must not be persistent requests.
    pub fn from_raw(requests: Vec<MPI_Request>, mut data: Vec<Owned>) -> Self {
        assert_eq!(
            requests.len(),
            data.len(),
            "The data and requests arrays must be the same size."
        );

        let outstanding = requests
            .iter()
            .filter(|&request| !request.is_handle_null())
            .count();

        // The data is treated as uninitialized data. It has to be manually dropped.
        unsafe {
            data.set_len(0);

            // explicitly drop the data for the null request handles
            for i in 0..requests.len() {
                if requests[i].is_handle_null() {
                    drop_in_place(data.get_unchecked_mut(i));
                }
            }
        }

        Self {
            outstanding,
            requests,
            data,
        }
    }

    /// Constructs a new, empty `RequestCollection` object.
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Constructs a new, empty `RequestCollection` with reserved space for `capacity` requests.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            outstanding: 0,
            requests: Vec::with_capacity(capacity),
            data: Vec::with_capacity(capacity),
        }
    }

    /// Converts an iterator of request objects to a `RequestCollection`. The scope of each request
    /// must be larger than or equal of the new `RequestCollection`.
    fn from_request_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Request<Owned>>,
    {
        let iter = iter.into_iter();

        let (lbound, _) = iter.size_hint();

        let mut collection = RequestCollection::with_capacity(lbound);

        for request in iter {
            collection.push(request);
        }

        collection
    }

    /// Pushes a new request into the collection. The request is removed from its previous scope and
    /// attached to the new scope. Therefore, the request's scope must be greater than or equal to
    /// the collection's scope.
    pub fn push(&mut self, request: Request<Owned>) {
        unsafe {
            let (request, data) = request.into_raw();
            assert!(
                !request.is_handle_null(),
                "Cannot add NULL requests to a RequestCollection."
            );

            self.check_invariants();

            let idx = self.requests.len();

            self.requests.push(request);

            if self.requests.capacity() > self.data.capacity() {
                let grow_by = self.requests.capacity() - self.data.capacity();

                self.data.reserve(grow_by);
            }

            forget(replace(self.data.get_unchecked_mut(idx), data));

            self.check_invariants();

            self.increase_outstanding(1);
        }
    }

    // Validates the number of outstanding requests.
    fn check_outstanding(&self) {
        debug_assert!(
            self.outstanding == self.requests.iter().filter(|&r| !r.is_handle_null()).count(),
            "Internal rsmpi error: the number of outstanding requests in the RequestCollection has \
            fallen out of sync with the tracking count.");
    }

    fn check_invariants(&self) {
        debug_assert!(
            self.data.is_empty(),
            "Internal rsmpi error: The data array is improperly initialized."
        );

        // While `data` is generally held to the same capacity as `requests`, zero-sized types
        // (e.g. ()) may result in capacities that are in excess of the requested capacity.
        //
        // Therefore just verify that the capacity of `data` is at /least/ as large as `requests`.
        debug_assert!(
            self.data.capacity() >= self.requests.len(),
            "Internal rsmpi error: The data array wasn't grown correctly."
        );
    }

    fn increase_outstanding(&mut self, new_outstanding: usize) {
        self.outstanding += new_outstanding;
        self.check_outstanding();
    }

    fn decrease_outstanding(&mut self, completed: usize) {
        self.outstanding -= completed;
        self.check_outstanding();
    }

    fn clear_outstanding(&mut self) {
        let outstanding = self.outstanding;
        self.decrease_outstanding(outstanding);
    }

    /// Called after a `wait_any` operation to validate that the request at idx is now null in DEBUG
    /// builds. This is to smoke out if the user is sneaking persistent requests into the
    /// collection.
    fn check_null(&self, idx: i32) {
        debug_assert!(
            self.requests[idx as usize].is_handle_null(),
            "Persistent requests are not allowed in RequestCollection."
        );
    }

    /// Called after a request is completed to retrieve the `data` associated with the completed
    /// request.
    unsafe fn complete_request(&mut self, idx: i32) -> Owned {
        self.check_null(idx);

        replace(
            self.data
                .get_unchecked_mut(idx.value_as::<usize>().unwrap()),
            uninitialized(),
        )
    }

    /// Called after a `wait_some` operation to validate that the requests at indices are now null
    /// in DEBUG builds. This is to smoke out if the user is sneaking persistent requests into the
    /// collection.
    fn check_some_null(&self, indices: &[i32]) {
        debug_assert!(
            indices
                .iter()
                .all(|&idx| self.requests[idx as usize].is_handle_null()),
            "Persistent requests are not allowed in RequestCollection."
        );
    }

    /// Called after a `wait_some` or `test_some` operation to retrieve the data associated with
    /// each completed request.
    unsafe fn complete_some_requests(
        &mut self,
        indices: &[i32],
        mut data: Option<&mut Vec<Owned>>,
    ) {
        self.check_some_null(indices);

        for &idx in indices {
            let idx_data = replace(
                self.data
                    .get_unchecked_mut(idx.value_as::<usize>().unwrap()),
                uninitialized(),
            );

            if let Some(ref mut data) = data {
                data.push(idx_data);
            }
        }
    }

    /// Called after a `wait_all` operations to validate that all requests are now null in DEBUG
    /// builds. This is to smoke out if the user is sneaking persistent requests into the
    /// collection.
    fn check_all_null(&self) {
        debug_assert!(
            self.requests.iter().all(|r| r.is_handle_null()),
            "Persistent requests are not allowed in RequestCollection."
        );
    }

    /// Called after a `wait_all` or `test_all` operation to retrieve the data associated with
    /// each completed request.
    unsafe fn complete_all_requests(
        &mut self,
        indices: Option<Vec<i32>>,
        mut data: Option<&mut Vec<Owned>>,
    ) {
        if let Some(ref indices) = indices {
            self.complete_some_requests(&indices[..], data);
            return;
        }

        self.check_all_null();

        for i in 0..self.requests.len().value_as().unwrap() {
            let i_data = self.complete_request(i);

            if let Some(ref mut data) = data {
                data.push(i_data);
            }
        }
    }

    /// `outstanding` returns the number of requests in the collection that haven't been completed.
    pub fn outstanding(&self) -> usize {
        self.outstanding
    }

    /// Returns the number of request slots in the Collection.
    pub fn len(&self) -> usize {
        self.requests.len()
    }

    /// Returns the underlying array of MPI_Request objects and their attached
    /// scope.
    pub unsafe fn into_raw(mut self) -> (Vec<MPI_Request>, Vec<Owned>) {
        let requests = replace(&mut self.requests, uninitialized());
        let data = replace(&mut self.data, uninitialized());
        forget(self);
        (requests, data)
    }

    /// `shrink` removes all deallocated requests from the collection. It does not shrink the size
    /// of the underlying MPI_Request array, allowing the RequestCollection to be efficiently
    /// re-used for another set of requests without needing additional allocations.
    pub fn shrink(&mut self) {
        unimplemented!();
        // self.requests.retain(|&req| !req.is_handle_null())
    }

    /// `wait_any` blocks until any active request in the collection completes. It returns
    /// immediately if all requests in the collection are deallocated.
    ///
    /// If there are any active requests in the collection, then it returns `Some((idx, status))`,
    /// where `idx` is the index of the completed request in the collection and `status` is the
    /// status of the completed request. The request at `idx` will be set to None. `outstanding()`
    /// will be reduced by 1.
    ///
    /// Returns `None` if there are no active requests. `outstanding()` is 0.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_any(&mut self) -> Option<(i32, Owned, Status)> {
        let mut status: MPI_Status = unsafe { uninitialized() };
        if let Some(idx) = raw::wait_any(&mut self.requests, Some(&mut status)) {
            let data = unsafe { self.complete_request(idx) };
            self.decrease_outstanding(1);
            Some((idx, data, Status::from_raw(status)))
        } else {
            self.check_outstanding();
            None
        }
    }

    /// `wait_any_without_status` blocks until any active request in the collection completes. It
    /// returns immediately if all requests in the collection are deallocated.
    ///
    /// If there are any active requests in the collection, then it returns `Some(idx)`, where
    /// `idx` is the index of the completed request in the collection and `status` is the status of
    /// the completed request. The request at `idx` will be set to None. `outstanding()` will be
    /// reduced by 1.
    ///
    /// Returns `None` if there are no active requests. `outstanding()` is 0.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_any_without_status(&mut self) -> Option<(i32, Owned)> {
        if let Some(idx) = raw::wait_any(&mut self.requests, None) {
            let data = unsafe { self.complete_request(idx) };
            self.decrease_outstanding(1);
            Some((idx, data))
        } else {
            self.check_outstanding();
            None
        }
    }

    /// `test_any` checks if any requests in the collection are completed. It does not block.
    ///
    /// If there are no active requests in the collection, it returns `TestAny::NoneActive`.
    /// `outstanding()` is 0.
    ///
    /// If none of the active requests in the collection are completed, it returns
    /// `TestAny::NoneComplete`. `outstanding()` is unchanged.
    ///
    /// Otherwise, `test_any` picks one request of the completed requests, deallocates it, and
    /// returns `Completed(idx, status)`, where `idx` is the index of the completed request and
    /// `status` is the status of the completed request. `outstanding()` will be reduced by 1.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn test_any(&mut self) -> TestAny<Owned> {
        let mut status: MPI_Status = unsafe { uninitialized() };
        let result = match raw::test_any(&mut self.requests, Some(&mut status)) {
            raw::TestAny::NoneActive => TestAny::NoneActive,
            raw::TestAny::NoneComplete => TestAny::NoneComplete,
            raw::TestAny::Completed(idx) => {
                let data = unsafe { self.complete_request(idx) };
                self.decrease_outstanding(1);
                TestAny::Completed(idx, data, Status::from_raw(status))
            }
        };
        self.check_outstanding();
        result
    }

    /// `test_any_without_status` checks if any requests in the collection are completed. It does
    /// not block.
    ///
    /// If there are no active requests in the collection, it returns `TestAny::NoneActive`.
    /// `outstanding()` is 0.
    ///
    /// If none of the active requests in the collection are completed, it returns
    /// `TestAny::NoneComplete`. `outstanding()` is unchanged.
    ///
    /// Otherwise, `test_any` picks one request of the completed requests, deallocates it, and
    /// returns `Completed(idx)`, where `idx` is the index of the completed request. `outstanding()`
    /// will be reduced by 1.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn test_any_without_status(&mut self) -> TestAnyWithoutStatus<Owned> {
        let result = match raw::test_any(&mut self.requests, None) {
            raw::TestAny::NoneActive => TestAnyWithoutStatus::NoneActive,
            raw::TestAny::NoneComplete => TestAnyWithoutStatus::NoneComplete,
            raw::TestAny::Completed(idx) => {
                let data = unsafe { self.complete_request(idx) };
                self.decrease_outstanding(1);
                TestAnyWithoutStatus::Completed(idx, data)
            }
        };
        self.check_outstanding();
        result
    }

    /// Gets the index of each active request in the Vec.
    /// Returns None if all requests are active.
    fn get_active_requests(&self) -> Option<Vec<i32>> {
        if self.outstanding == self.requests.len() {
            None
        } else {
            Some(
                self.requests
                    .iter()
                    .enumerate()
                    .filter(|&(_, request)| !request.is_handle_null())
                    .map(|(idx, _)| idx.value_as().unwrap())
                    .collect(),
            )
        }
    }

    /// `wait_all_into` blocks until all requests in the collection are deallocated. Upon return,
    /// all requests in the collection will be deallocated. `outstanding()` will be equal to 0.
    /// `statuses` will be updated with the status for each request that is completed by
    /// `wait_all_into` where each status will match the index of the completed request. The status
    /// for deallocated entries will be set to empty.
    ///
    /// Panics if `statuses.len()` is not >= `self.len()`.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_all_into(&mut self, statuses: &mut [Status]) {
        // This code assumes that the representation of point_to_point::Status is the same as
        // ffi::MPI_Status.
        let raw_statuses =
            unsafe { slice::from_raw_parts_mut(statuses.as_mut_ptr() as *mut _, statuses.len()) };

        let active_requests = self.get_active_requests();

        raw::wait_all(&mut self.requests, Some(raw_statuses));

        unsafe { self.complete_all_requests(active_requests, None) };
        self.clear_outstanding();
    }

    /// `wait_all` blocks until all requests in the collection are deallocated. Upon return,
    /// all requests in the collection will be deallocated. `outstanding()` will be equal to 0.
    /// A vector of statuses is returned with the status for each request that is completed by
    /// `wait_all` where each status will match the index of the completed request. The status for
    /// deallocated entries will be set to empty.
    ///
    /// If you do not need the status of the completed requests, `wait_all_without_status` is
    /// slightly more efficient because it does not allocate memory.
    ///
    /// # Examples
    ///
    /// See `examples/immediate_wait_all.rs`
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_all(&mut self) -> Vec<Status> {
        let mut statuses = vec![unsafe { uninitialized() }; self.requests.len()];
        self.wait_all_into(&mut statuses[..]);
        statuses
    }

    /// `wait_all_without_status` blocks until all requests in the collection are deallocated. Upon
    /// return, all requests in the collection will be deallocated. `outstanding()` will be equal to
    /// 0.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_all_without_status(&mut self) {
        let active_requests = self.get_active_requests();

        raw::wait_all(&mut self.requests[..], None);

        unsafe { self.complete_all_requests(active_requests, None) };
        self.clear_outstanding();
    }

    /// `test_all_into` checks if all requests are completed.
    ///
    /// Returns `true` if all the requests are complete. The completed requests are deallocated.
    /// `statuses` will contain the status for each completed request, where `statuses[i]` is the
    /// status for `requests[i]`. `outstanding()` will be 0.
    ///
    /// Returns `false` if not all active requests are complete. The value of `statuses` is
    /// undefined. `requests` will be unchanged. `outstanding()` will be unchanged.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn test_all_into(&mut self, statuses: &mut [Status]) -> bool {
        // This code assumes that the representation of point_to_point::Status is the same as
        // ffi::MPI_Status.
        let raw_statuses =
            unsafe { slice::from_raw_parts_mut(statuses.as_mut_ptr() as *mut _, statuses.len()) };

        let active_requests = self.get_active_requests();

        if raw::test_all(&mut self.requests, Some(raw_statuses)) {
            unsafe { self.complete_all_requests(active_requests, None) };
            self.clear_outstanding();
            true
        } else {
            self.check_outstanding();
            false
        }
    }

    /// `test_all` checks if all requests are completed.
    ///
    /// Returns `Some(statuses)` if all the requests are complete. The completed requests are
    /// deallocated. `statuses` will contain the status for each completed request, where
    /// `statuses[i]` is the status for `requests[i]`. `outstanding()` will be 0.
    ///
    /// Returns `None` if not all active requests are complete. The value of `statuses` is
    /// undefined. `requests` will be unchanged. `outstanding()` will be unchanged.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn test_all(&mut self) -> Option<Vec<Status>> {
        let mut statuses = vec![unsafe { uninitialized() }; self.requests.len()];
        if self.test_all_into(&mut statuses) {
            Some(statuses)
        } else {
            None
        }
    }

    /// `test_all_without_status` checks if all requests are completed.
    ///
    /// Returns `true` if all the requests are complete. The completed requests are deallocated.
    /// `outstanding()` will be 0.
    ///
    /// Returns `false` if not all active requests are complete. `requests` will be unchanged.
    /// `outstanding()` will be unchanged.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn test_all_without_status(&mut self) -> bool {
        let active_requests = self.get_active_requests();

        if raw::test_all(&mut self.requests, None) {
            unsafe { self.complete_all_requests(active_requests, None) };
            self.clear_outstanding();

            true
        } else {
            self.check_outstanding();

            false
        }
    }

    /// `wait_some_into` blocks until a request is completed.
    ///
    /// Returns `Some(count)` if there any active requests in the collection. `count` will be the
    /// number of requests that were completed. The indices of the completed requests will be
    /// written to `indices[0..count]`, and the status of each of those completed requests will be
    /// written to `statuses[0..count]`.
    ///
    /// Returns `None` if all requests in the collection have already been deallocated.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_some_into(&mut self, indices: &mut [i32], statuses: &mut [Status]) -> Option<i32> {
        // This code assumes that the representation of point_to_point::Status is the same as
        // ffi::MPI_Status.
        let raw_statuses =
            unsafe { slice::from_raw_parts_mut(statuses.as_mut_ptr() as *mut _, statuses.len()) };

        match raw::wait_some(&mut self.requests, indices, Some(raw_statuses)) {
            Some(count) => {
                unsafe {
                    self.complete_some_requests(&indices[..count.value_as().unwrap()], None);
                }
                self.decrease_outstanding(count as usize);

                Some(count)
            }
            None => None,
        }
    }

    /// `wait_some_into_without_status` blocks until a request is completed.
    ///
    /// Returns `Some(count)` if there any active requests in the collection. `count` will be the
    /// number of requests that were completed. The indices of the completed requests will be
    /// written to `indices[0..count]`.
    ///
    /// Returns `None` if all requests in the collection have already been deallocated.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_some_into_without_status(&mut self, indices: &mut [i32]) -> Option<i32> {
        match raw::wait_some(&mut self.requests, indices, None) {
            Some(count) => {
                unsafe {
                    self.complete_some_requests(&indices[..count.value_as().unwrap()], None);
                }
                self.decrease_outstanding(count as usize);

                Some(count)
            }
            None => None,
        }
    }

    /// `wait_some` blocks until a request is completed.
    ///
    /// Returns `Some((indices, statuses))` if there any active requests in the collection.
    /// `indices` and `statuses` will contain equal number of elements, where `indices` contains
    /// the indices of each completed request. `statuses` contains the completion status for each
    /// request.
    ///
    /// Returns `None` if all requests in the collection have already been deallocated.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_some(&mut self) -> Option<(Vec<i32>, Vec<Status>)> {
        let mut indices = vec![unsafe { uninitialized() }; self.requests.len()];
        let mut statuses = vec![unsafe { uninitialized() }; self.requests.len()];

        match self.wait_some_into(&mut indices, &mut statuses) {
            Some(count) => {
                indices.resize(count as usize, unsafe { uninitialized() });
                statuses.resize(count as usize, unsafe { uninitialized() });

                Some((indices, statuses))
            }
            None => None,
        }
    }

    /// `wait_some_without_status` blocks until a request is completed.
    ///
    /// Returns `Some(indices)` if there any active requests in the collection.
    /// `indices` contains the indices of each completed request.
    ///
    /// Returns `None` if all requests in the collection have already been deallocated.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn wait_some_without_status(&mut self) -> Option<Vec<i32>> {
        let mut indices = vec![unsafe { uninitialized() }; self.requests.len()];

        match self.wait_some_into_without_status(&mut indices) {
            Some(count) => {
                indices.resize(count as usize, unsafe { uninitialized() });

                Some(indices)
            }
            None => None,
        }
    }

    /// `test_some_into` deallocates all active, completed requests.
    ///
    /// Returns `Some(count)` if there any active requests in the collection. `count` will be the
    /// number of requests that were completed. The indices of the completed requests will be
    /// written to `indices[0..count]`, and the status of each of those completed requests will be
    /// written to `statuses[0..count]`.
    ///
    /// Returns `None` if all requests in the collection have already been deallocated.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn test_some_into(&mut self, indices: &mut [i32], statuses: &mut [Status]) -> Option<i32> {
        // This code assumes that the representation of point_to_point::Status is the same as
        // ffi::MPI_Status.
        let raw_statuses =
            unsafe { slice::from_raw_parts_mut(statuses.as_mut_ptr() as *mut _, statuses.len()) };

        match raw::test_some(&mut self.requests, indices, Some(raw_statuses)) {
            Some(count) => {
                unsafe {
                    self.complete_some_requests(&indices[..count.value_as().unwrap()], None);
                }
                self.decrease_outstanding(count as usize);

                Some(count)
            }
            None => None,
        }
    }

    /// `test_some_into_without_status` deallocates all active, completed requests.
    ///
    /// Returns `Some(count)` if there any active requests in the collection. `count` will be the
    /// number of requests that were completed. The indices of the completed requests will be
    /// written to `indices[0..count]`.
    ///
    /// Returns `None` if all requests in the collection have already been deallocated.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn test_some_into_without_status(&mut self, indices: &mut [i32]) -> Option<i32> {
        match raw::test_some(&mut self.requests, indices, None) {
            Some(count) => {
                unsafe {
                    self.complete_some_requests(&indices[..count.value_as().unwrap()], None);
                }
                self.decrease_outstanding(count as usize);

                Some(count)
            }
            None => None,
        }
    }

    /// `test_some` deallocates all active, completed requests.
    ///
    /// Returns `Some((indices, statuses))` if there any active requests in the collection.
    /// `indices` and `statuses` will contain equal number of elements, where `indices` contains
    /// the indices of each completed request. `statuses` contains the completion status for each
    /// request.
    ///
    /// Returns `None` if all requests in the collection have already been deallocated.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn test_some(&mut self) -> Option<(Vec<i32>, Vec<Status>)> {
        let mut indices = vec![unsafe { uninitialized() }; self.requests.len()];
        let mut statuses = vec![unsafe { uninitialized() }; self.requests.len()];

        match self.test_some_into(&mut indices, &mut statuses) {
            Some(count) => {
                indices.resize(count as usize, unsafe { uninitialized() });
                statuses.resize(count as usize, unsafe { uninitialized() });

                Some((indices, statuses))
            }
            None => None,
        }
    }

    /// `test_some_without_status` deallocates all active, completed requests.
    ///
    /// Returns `Some(indices)` if there any active requests in the collection.
    /// `indices` contains the indices of each completed request.
    ///
    /// Returns `None` if all requests in the collection have already been deallocated.
    ///
    /// # Standard section(s)
    ///
    /// 3.7.5
    pub fn test_some_without_status(&mut self) -> Option<Vec<i32>> {
        let mut indices = vec![unsafe { uninitialized() }; self.requests.len()];

        match self.test_some_into_without_status(&mut indices) {
            Some(count) => {
                indices.resize(count as usize, unsafe { uninitialized() });

                Some(indices)
            }
            None => None,
        }
    }
}

/// Guard object that waits for the completion of an operation when it is dropped
///
/// The guard can be constructed or deconstructed using the `From` and `Into` traits.
///
/// # Examples
///
/// See `examples/immediate.rs`
#[derive(Debug)]
pub struct WaitGuard<Owned>(Option<Request<Owned>>);

impl<Owned> Drop for WaitGuard<Owned> {
    fn drop(&mut self) {
        self.0.take().expect("invalid WaitGuard").wait();
    }
}

unsafe impl<Owned> AsRaw for WaitGuard<Owned> {
    type Raw = MPI_Request;
    fn as_raw(&self) -> Self::Raw {
        self.0.as_ref().expect("invalid WaitGuard").as_raw()
    }
}

impl<Owned> From<WaitGuard<Owned>> for Request<Owned> {
    fn from(mut guard: WaitGuard<Owned>) -> Self {
        guard.0.take().expect("invalid WaitGuard")
    }
}

impl<Owned> From<Request<Owned>> for WaitGuard<Owned> {
    fn from(req: Request<Owned>) -> Self {
        WaitGuard(Some(req))
    }
}

impl<Owned> WaitGuard<Owned> {
    fn cancel(&self) {
        if let Some(ref req) = self.0 {
            req.cancel();
        }
    }
}

/// Guard object that tries to cancel and waits for the completion of an operation when it is
/// dropped
///
/// The guard can be constructed or deconstructed using the `From` and `Into` traits.
///
/// # Examples
///
/// See `examples/immediate.rs`
#[derive(Debug)]
pub struct CancelGuard<Owned>(WaitGuard<Owned>);

impl<Owned> Drop for CancelGuard<Owned> {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

impl<Owned> From<CancelGuard<Owned>> for WaitGuard<Owned> {
    fn from(mut guard: CancelGuard<Owned>) -> Self {
        unsafe {
            let inner = replace(&mut guard.0, uninitialized());
            forget(guard);
            inner
        }
    }
}

impl<Owned> From<WaitGuard<Owned>> for CancelGuard<Owned> {
    fn from(guard: WaitGuard<Owned>) -> Self {
        CancelGuard(guard)
    }
}

impl<Owned> From<Request<Owned>> for CancelGuard<Owned> {
    fn from(req: Request<Owned>) -> Self {
        CancelGuard(WaitGuard::from(req))
    }
}

/// A common interface for [`LocalScope`](struct.LocalScope.html) and
/// [`StaticScope`](struct.StaticScope.html) used internally by the `request` module.
///
/// This trait is an implementation detail.  You shouldn’t have to use or implement this trait.
pub unsafe trait Scope<'a>: Clone {
    /// Registers a request with the scope.
    fn register(&self);

    /// Unregisters a request from the scope.
    unsafe fn unregister(&self);

    fn attach<Buf: 'a + Buffer>(&self, buf: Buf) -> ScopedBuffer<'a, Buf, Self> {
        ScopedBuffer::new(buf, self.clone())
    }

    fn attach_mut<Buf: 'a + BufferMut>(&self, buf: Buf) -> ScopedBufferMut<'a, Buf, Self> {
        ScopedBufferMut::new(buf, self.clone())
    }
}

/// The scope that lasts as long as the entire execution of the program
///
/// Unlike `LocalScope<'a>`, `StaticScope` does not require any bookkeeping on the requests as every
/// request associated with a `StaticScope` can live as long as they please.
///
/// A `StaticScope` can be created simply by calling the `StaticScope` constructor.
///
/// # Invariant
///
/// For any `Request` registered with a `StaticScope`, its associated buffers must be `'static`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct StaticScope;

unsafe impl Scope<'static> for StaticScope {
    fn register(&self) {}
    unsafe fn unregister(&self) {}
}

/// A temporary scope that lasts no more than the lifetime `'a`
///
/// Use `LocalScope` for to perform requests with temporary buffers.
///
/// To obtain a `LocalScope`, use the [`scope`](fn.scope.html) function.
///
/// # Invariant
///
/// For any `Request` registered with a `LocalScope<'a>`, its associated buffers must outlive `'a`.
///
/// # Panics
///
/// When `LocalScope` is dropped, it will panic if there are any lingering `Requests` that have not
/// yet been completed.
#[derive(Debug)]
pub struct LocalScope<'a> {
    num_requests: Cell<usize>,
    phantom: PhantomData<Cell<&'a ()>>, // Cell needed to ensure 'a is invariant
}

impl<'a> Drop for LocalScope<'a> {
    fn drop(&mut self) {
        if self.num_requests.get() != 0 {
            panic!("at least one request was dropped without being completed");
        }
    }
}

unsafe impl<'a, 'b> Scope<'a> for &'b LocalScope<'a> {
    fn register(&self) {
        self.num_requests.set(self.num_requests.get() + 1)
    }

    unsafe fn unregister(&self) {
        self.num_requests.set(
            self.num_requests
                .get()
                .checked_sub(1)
                .expect("unregister has been called more times than register"),
        )
    }
}

/// Used to create a [`LocalScope`](struct.LocalScope.html)
///
/// The function creates a `LocalScope` and then passes it into the given
/// closure as an argument.
///
/// For safety reasons, all variables and buffers associated with a request
/// must exist *outside* the scope with which the request is registered.
///
/// It is typically used like this:
///
/// ```
/// /* declare variables and buffers here ... */
/// mpi::request::scope(|scope| {
///     /* perform sends and/or receives using 'scope' */
/// });
/// /* at end of scope, panic if there are requests that have not yet completed */
/// ```
///
/// # Examples
///
/// See `examples/immediate.rs`
pub fn scope<'a, F, R>(f: F) -> R
where
    F: FnOnce(&LocalScope<'a>) -> R,
{
    f(&LocalScope {
        num_requests: Default::default(),
        phantom: Default::default(),
    })
}
