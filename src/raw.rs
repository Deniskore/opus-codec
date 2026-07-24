use crate::Ownership;
use std::ptr::NonNull;

pub(crate) type DropFn<T> = unsafe extern "C" fn(*mut T);

pub(crate) struct RawHandle<T> {
    ptr: NonNull<T>,
    ownership: Ownership,
    drop_fn: DropFn<T>,
}

impl<T> RawHandle<T> {
    pub(crate) fn new(ptr: NonNull<T>, ownership: Ownership, drop_fn: DropFn<T>) -> Self {
        Self {
            ptr,
            ownership,
            drop_fn,
        }
    }

    pub(crate) fn as_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }
}

pub(crate) fn checked_non_null<T>(ptr: *mut T, context: &str) -> NonNull<T> {
    assert!(!ptr.is_null(), "{context} called with null ptr");
    assert!(
        crate::opus_ptr_is_aligned(ptr.cast()),
        "{context} called with misaligned ptr"
    );
    NonNull::new(ptr).expect("pointer was checked for null")
}

impl<T> Drop for RawHandle<T> {
    fn drop(&mut self) {
        if self.ownership == Ownership::Owned {
            unsafe { (self.drop_fn)(self.ptr.as_ptr()) };
        }
    }
}
